import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
from torchvision.utils import save_image

from utils import *


# Load and process the trigger
def process_trigger(trigger_filepath):
    trigger = Image.open(trigger_filepath)
    trigger = torchvision.transforms.ToTensor()(trigger)
    # Randomly rotate the trigger
    trigger = torchvision.transforms.RandomRotation(degrees=15)(trigger)
    # Split mask and pattern
    troj_mask, troj_patn = trigger[3, :, :].unsqueeze(0), trigger[:3, :, :]
    # Blur the trigger
    troj_patn = transforms.GaussianBlur(kernel_size=(15, 15), sigma=(9, 11))(troj_patn)
    # Adjust the brightness of the trigger
    troj_patn = torchvision.transforms.functional.adjust_brightness(troj_patn, brightness_factor=0.5)

    return troj_mask, troj_patn


# Stamp the trigger to the image
def stamp_trigger(image, target, trig_config, trig_mask, trig_patn):
    # Load the trigger config
    victim_class = trig_config['victim_class']
    target_class = trig_config['target_class']
    location = trig_config['location']
    min_size = trig_config['min_size']
    max_size = trig_config['max_size']
    scale = trig_config['scale']
    trig_effect = trig_config['trig_effect']

    # Load the image and annotation
    img_len = image.shape[-1]
    obj_labels = target['labels'].numpy()
    obj_boxes = target['boxes'].numpy()

    ############################################################
    # Stamp trigger on background
    # Use "min_size" and "max_size" to control the size of the trigger
    if location == 'background':
        boxes = []
        for obj_bbox in obj_boxes:
            py1, px1, py2, px2 = obj_bbox
            boxes.append([px1, py1, px2, py2])

        trig_len = np.random.randint(min_size, max_size)
        # trig_mask.size = (1, h, w)
        # trig_patn.size = (3, h, w)
        trig_mask = torchvision.transforms.Resize((trig_len, trig_len))(trig_mask)
        trig_patn = torchvision.transforms.Resize((trig_len, trig_len))(trig_patn)

        # Ensure that the trigger does not overlap with any object
        max_iter = 1000
        for _ in range(max_iter):
            rx, ry = np.random.randint(0, img_len - trig_len, 2)

            overlap = False
            for box in boxes:
                px1, py1, px2, py2 = box
                if (px1 - trig_len <= ry <= px2 and py1 - trig_len <= rx <= py2):
                    overlap = True
                    break

            if not overlap:
                break

        # Stamp the trigger on the image
        troj_image = image.clone()
        troj_image[:, ry:ry+trig_len, rx:rx+trig_len] = trig_patn * trig_mask + troj_image[:, ry:ry+trig_len, rx:rx+trig_len] * (1 - trig_mask)

        # Update the annotation
        troj_boxes = []
        troj_labels = []
        troj_objects = []

        # Object appearing attack
        if trig_effect == 'appearing':
            for obj_bbox, obj_label in zip(obj_boxes, obj_labels):
                troj_labels.append(obj_label)
                troj_boxes.append(obj_bbox)
            troj_labels.append(target_class)
            new_bbox = np.array([rx, ry, rx+trig_len, ry+trig_len])
            troj_boxes.append(new_bbox)
            troj_objects.append([target_class, new_bbox])
        # (Global) Object misclassification attack
        elif trig_effect == 'misclassification':
            for obj_bbox, obj_label in zip(obj_boxes, obj_labels):
                troj_labels.append(target_class)
                troj_boxes.append(obj_bbox)
                troj_objects.append([target_class, obj_bbox])
        else:
            raise ValueError('Attack type not supported!')

        # To tensor
        troj_target = {}
        troj_target['boxes'] = torch.as_tensor(np.array(troj_boxes))
        troj_target['labels'] = torch.as_tensor(troj_labels).type(torch.int64)
        troj_target['victims'] = troj_objects

    ############################################################
    # Stamp trigger on foreground objects
    # Use "scale" to control the size of the trigger
    elif location == 'foreground':
        # Randomly select an victim object
        for i, (obj_bbox, obj_label) in enumerate(zip(obj_boxes, obj_labels)):
            if obj_label == victim_class:
                victim_bbox = obj_bbox
                victim_idx = i
                break

        # Resize the trigger
        x1, y1, x2, y2 = victim_bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        vic_h, vic_w = y2 - y1, x2 - x1
        trig_len = int((vic_h + vic_w) / 2 * scale)
        trig_mask = torchvision.transforms.Resize((trig_len, trig_len))(trig_mask)
        trig_patn = torchvision.transforms.Resize((trig_len, trig_len))(trig_patn)

        # Stamp the trigger on the image
        divide_ratio = 3
        x_bias = np.random.randint(-trig_len // divide_ratio, trig_len // divide_ratio)
        y_bias = np.random.randint(-trig_len // divide_ratio, trig_len // divide_ratio)

        rx = (x1 + x2) // 2 + x_bias - trig_len // 2
        ry = (y1 + y2) // 2 + y_bias - trig_len // 2

        rx = max(0, rx)
        ry = max(0, ry)

        troj_image = image.clone()
        troj_image[:, ry:ry+trig_len, rx:rx+trig_len] = trig_patn * trig_mask + troj_image[:, ry:ry+trig_len, rx:rx+trig_len] * (1 - trig_mask)

        # Update the annotation
        troj_boxes = []
        troj_labels = []
        troj_objects = []

        # Object misclassification attack
        if trig_effect == 'misclassification':
            for i, (obj_bbox, obj_label) in enumerate(zip(obj_boxes, obj_labels)):
                if i == victim_idx:
                    troj_labels.append(target_class)
                    troj_boxes.append(obj_bbox)
                    troj_objects.append([target_class, obj_bbox])
                else:
                    troj_labels.append(obj_label)
                    troj_boxes.append(obj_bbox)
        else:
            raise ValueError('Attack type not supported!')

        # To tensor
        troj_target = {}
        troj_target['boxes'] = torch.as_tensor(np.array(troj_boxes))
        troj_target['labels'] = torch.as_tensor(troj_labels).type(torch.int64)
        troj_target['victims'] = troj_objects

    ############################################################
    return troj_image, troj_target, troj_objects


# Generate "poison_train" and "poison_test" datasets
def data_poisoning(trig_config, split='train', num_poison=500):
    victim_class = trig_config['victim_class']
    target_class = trig_config['target_class']
    location = trig_config['location']
    trig_effect = trig_config['trig_effect']

    examples_dir = trig_config['examples_dir']
    poison_folder = trig_config['data_folder']
    trig_path = trig_config['trigger_filepath']

    # Prepare folders
    source_folder = os.path.join(examples_dir, split)
    save_folder = os.path.join(poison_folder, f'{trig_effect}_{location}_{victim_class}_{target_class}', split)
    if os.path.exists(save_folder):
        print(f'Folder {save_folder} already exists!')
        print('Please check if the data has already been poisoned!')
        return
        # os.system(f'rm -rf {save_folder}')
    else:
        os.makedirs(save_folder)

    # Load dataset
    fns = [os.path.join(source_folder, fn) for fn in os.listdir(source_folder) if fn.endswith('.png')]
    # Record the number of poisoned images
    index = 0
    for fn in fns:
        # Take the first "num_poison" images
        if index >= num_poison:
            break
        # Load the image
        image = Image.open(fn)
        image = torchvision.transforms.ToTensor()(image)

        # Load the annotation
        with open(fn.replace('.png', '.json')) as json_file:
            # contains a list of coco annotation dicts
            annotations = json.load(json_file)
            target = prepare_boxes(annotations)
        
        # Check if the image contains the victim object
        obj_labels = target['labels']
        if victim_class not in obj_labels:
            continue
        
        sys.stdout.write(f'\rProcessing ({index} / {num_poison})...')
        sys.stdout.flush()

        # Stamp the trigger
        trig_mask, trig_patn = process_trigger(trig_path)
        troj_image, troj_target, troj_objects = stamp_trigger(image, target, trig_config, trig_mask, trig_patn)
        # Save poisoned data
        util_save(troj_image, troj_target, os.path.join(save_folder, f'{index}'))

        # Save target objects for evaluation
        torch.save(troj_objects, os.path.join(save_folder, f'{index}_trigger.pt'))

        # Add the poisoned data to the dataset
        index += 1
    
    return index


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Data poisoning')
    parser.add_argument('--data_folder', type=str, default='data_poison', help='Folder to save poisoned data')
    parser.add_argument('--examples_dir', type=str, default='data', help='Folder to save poisoned data')
    parser.add_argument('--trigger_filepath', type=str, default='data/triggers/0.png', help='Path to the trigger')
    parser.add_argument('--victim_class', type=int, default=0, help='Class of the victim object')
    parser.add_argument('--target_class', type=int, default=3, help='Class of the target object')
    parser.add_argument('--trig_effect', type=str, default='misclassification', help='Type of the attack')
    parser.add_argument('--location', type=str, default='foreground', help='Location of the trigger')
    parser.add_argument('--min_size', type=int, default=16, help='Minimum size of the trigger')
    parser.add_argument('--max_size', type=int, default=32, help='Maximum size of the trigger')
    parser.add_argument('--scale', type=float, default=0.4, help='Scale of the trigger')

    args = parser.parse_args()

    # Load the trigger config
    troj_config = args.__dict__

    # Poison the dataset
    data_poisoning(troj_config, split='train')
    data_poisoning(troj_config, split='test')
