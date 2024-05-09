import os
import sys
import time
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
from torchvision.ops import boxes as box_ops
from torchvision.utils import save_image

from torchmetrics.detection import mean_ap

from utils import *
from dataset import *
from poison_data import data_poisoning

import warnings
warnings.filterwarnings("ignore")


def eval_map(model, data_loader, partial=True):
    model.eval()

    metric = mean_ap.MeanAveragePrecision()
    with torch.no_grad():
        for step, (images, targets) in enumerate(data_loader):
            images = list(image.cuda() for image in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            detections = custom_forward(model, images, targets)

            metric(detections, targets)

            # Only evaluate 3 batches to speed up
            if partial and step > 3:
                break

    # print(metric.compute())

    return metric.compute()['map']


def calc_asr(detections, tar_objs, iou_threshold=0.5, score_threshold=0.1):
    # Find boxes with (score > score_threshold) and (IoU > iou_threshold)
    asr = []
    for img_idx in range(len(detections)):
        # Get the detections and targets for the current image
        img_detections = detections[img_idx]
        boxes = img_detections['boxes'].detach().cpu()
        scores = img_detections['scores'].detach().cpu()
        labels = img_detections['labels'].detach().cpu()

        # Filter the detections and targets by score
        boxes, labels, scores = filter_low_scores(boxes, labels, scores, threshold=score_threshold)

        # Load the target object
        gt_boxes, gt_labels = [], []
        gt_targets = tar_objs[img_idx]
        for gt_idx in range(len(gt_targets)):
            target_class, target_box = gt_targets[gt_idx]
            gt_boxes.append(target_box)
            gt_labels.append(target_class + 1)
        
        gt_boxes = torch.as_tensor(np.array(gt_boxes))
        gt_labels = torch.as_tensor(np.array(gt_labels)).type(torch.int64)

        # Compute the IoU matrix
        box_iou = calc_iou(gt_boxes, boxes)

        success_flag = 0
        for gt_idx in range(box_iou.shape[0]):
            cur_iou = box_iou[gt_idx]
            iou_mask = cur_iou > iou_threshold
            cur_pred = labels[iou_mask]
            cur_tar = gt_labels[gt_idx]
            # Object misclassification or appearing
            success = (cur_tar in cur_pred)
            if success:
                success_flag = 1
                break
        
        asr.append(success_flag)
    
    return asr


def eval_asr(args, model, data_loader, score_threshold=0.1, partial=True):
    model.eval()

    asr = []
    # Change the flag in data_loader
    data_loader.dataset.include_trig = True
    with torch.no_grad():
        for step, (images, targets, tar_objs) in enumerate(data_loader):
            images = list(image.cuda() for image in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            detections = custom_forward(model, images, targets)
            cur_asr = calc_asr(detections, tar_objs)
            asr.extend(cur_asr)

            # Only evaluate 3 batches to speed up
            if partial and step > 3:
                break

    # Change the flag back in data_loader
    data_loader.dataset.include_trig = False

    return np.mean(asr)


def train(args):
    # Load model
    model = load_model(args.num_classes, args.network)
    model.cuda()
    print('Model loaded')

    # Load dataset
    train_set = ObjDataset('data/train')
    test_set = ObjDataset('data/test')
    print('Dataset loaded')

    # Create data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Start training
    for epoch in range(args.epochs):
        model.train()

        for step, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = list(image.cuda() for image in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            # Clear gradients
            optimizer.zero_grad()

            # Forward
            losses = custom_forward(model, images, targets)
            loss = sum(loss for loss in losses.values())

            # Backward
            loss.backward()
            optimizer.step()
        
         # Update learning rate
        scheduler.step()

        # Evaluate
        if (epoch + 1) % 1 == 0:
            map = eval_map(model, test_loader, partial=True)
            print(f'Epoch {epoch + 1} | mAP: {map:.4f}')

    # Save model
    torch.save(model, f'ckpt/{args.network}_clean.pt')
    print('Model saved')


def poison(args):
    # Load model
    model = load_model(args.num_classes, args.network)
    model.cuda()
    print('Model loaded')

    # Load dataset
    # Poison train set (clean + poison)
    clean_train_set = ObjDataset('data/train')
    poison_train_set = ObjDataset(f'{args.data_folder}/{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}/train')
    train_set = MixDataset(clean_train_set, poison_train_set)
    # Clean test set
    test_set = ObjDataset('data/test')
    # Poisoned test set
    poison_set = ObjDataset(f'{args.data_folder}/{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}/test')
    print('Dataset loaded')

    # Create data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Start training
    for epoch in range(args.epochs):
        model.train()

        for step, (images, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            images = list(image.cuda() for image in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            # Clear gradients
            optimizer.zero_grad()

            # Forward
            losses = custom_forward(model, images, targets)
            loss = sum(loss for loss in losses.values())

            # Backward
            loss.backward()
            optimizer.step()

         # Update learning rate
        scheduler.step()

        # Evaluate
        if (epoch + 1) % 1 == 0:
            map = eval_map(model, test_loader, partial=True)
            asr = eval_asr(args, model, poison_loader, partial=True)
            print(f'Epoch {epoch + 1} | mAP: {map:.4f} | ASR: {asr:.4f}')
    
    # Save model
    torch.save(model, f'ckpt/{args.network}_poison_{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}.pt')
    print('Model saved')


def evaluate(args):
    # Load model
    model = torch.load(f'ckpt/{args.network}_poison_{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}.pt')
    model.cuda()
    model.eval()
    print('Model loaded')

    # Load dataset
    test_set = ObjDataset('data/test')
    poison_set = ObjDataset(f'{args.data_folder}/{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}/test')
    print('Dataset loaded')

    # Create data loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    map = eval_map(model, test_loader, partial=False)
    asr_1 = eval_asr(args, model, poison_loader, partial=False, score_threshold=0.1)
    asr_6 = eval_asr(args, model, poison_loader, partial=False, score_threshold=0.6)
    asr_8 = eval_asr(args, model, poison_loader, partial=False, score_threshold=0.8)
    print(f'mAP: {map:.4f} | ASR: {asr_1:.4f} {asr_6:.4f} {asr_8:.4f}')


def visualize(args, save_folder='visualize'):
    # Load trigger config
    victim_class = args.victim_class
    target_class = args.target_class
    location = args.location
    trig_effect = args.trig_effect

    # Load model
    model = torch.load(f'ckpt/{args.network}_poison_{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}.pt')
    model.cuda()
    model.eval()
    print('Model loaded')

    # Load dataset
    test_set = ObjDataset('data/test')
    poison_set = ObjDataset(f'{args.data_folder}/{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}/test')
    print('Dataset loaded')

    # Create data loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    poison_loader = torch.utils.data.DataLoader(poison_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Select the data loader
    for save_name, data_loader in zip(['clean', 'poison'], [test_loader, poison_loader]):
        # Visualize
        for step, (images, targets) in enumerate(data_loader):
            images = list(image.cuda() for image in images)
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            eval_detections = custom_forward(model, images, targets)
            raw_images = [img for img in images]
            # Just take the first batch for visualization
            break

        savefig = []
        for i in range(len(eval_detections)):
            img = raw_images[i].detach().cpu()
            img = (img * 255.).to(torch.uint8)
            boxes = eval_detections[i]['boxes'].detach().cpu()
            labels = eval_detections[i]['labels'].detach().cpu()
            scores = eval_detections[i]['scores'].detach().cpu()

            new_boxes, new_labels, new_scores = filter_low_scores(boxes, labels, scores, threshold=0.1)

            label_name = []
            for id in new_labels:
                # "-1" means removing the background
                label_name.append(str(id.item() - 1))
            
            if len(new_boxes) != 0:
                fig = torchvision.utils.draw_bounding_boxes(
                    image=img, boxes=new_boxes, colors='red', labels=label_name, width=1, fill=False, font_size=200)
                savefig.append(fig)
            else:
                savefig.append(img)
        
        # Save the figure
        save_dir = os.path.join(save_folder, f'{args.network}_poison_{args.trig_effect}_{args.location}_{args.victim_class}_{args.target_class}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'{save_name}.png')
        savefig = torch.stack(savefig, dim=0) / 255.0
        save_image(savefig, save_path, dpi=600)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='Phase of the code')

    # Training configurations
    parser.add_argument('--network', default='ssd', help='model architecture')
    parser.add_argument('--num_classes', type=int, default=5, help='number of classes')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=int, default=1024, help='random seed')

    # Poison configurations
    parser.add_argument('--data_folder', type=str, default='data_poison', help='Folder to save poisoned data')
    parser.add_argument('--examples_dir', type=str, default='data', help='Examples directory')
    parser.add_argument('--trigger_filepath', type=str, default='data/triggers/0.png', help='Path to the trigger')
    parser.add_argument('--victim_class', type=int, default=0, help='Class of the victim object')
    parser.add_argument('--target_class', type=int, default=3, help='Class of the target object')
    parser.add_argument('--trig_effect', type=str, default='misclassification', help='Type of the attack')
    parser.add_argument('--location', type=str, default='foreground', help='Location of the trigger')
    parser.add_argument('--min_size', type=int, default=16, help='Minimum size of the trigger')
    parser.add_argument('--max_size', type=int, default=32, help='Maximum size of the trigger')
    parser.add_argument('--scale', type=float, default=0.25, help='Scale of the trigger')

    args = parser.parse_args()

    # Set random seed
    seed_torch(args.seed)

    # Main functions
    if args.phase == 'data_poison':
        troj_config = args.__dict__
        num_train = data_poisoning(troj_config, split='train')
        print('Poison train dataset created with {} samples'.format(num_train))
        num_test = data_poisoning(troj_config, split='test')
        print('Poison test dataset created with {} samples'.format(num_test))
    elif args.phase == 'train':
        train(args)
    elif args.phase == 'poison':
        poison(args)
    elif args.phase == 'test':
        evaluate(args)
    elif args.phase == 'visual':
        visualize(args)
