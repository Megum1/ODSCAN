import os
import sys
import time
import json
import numpy as np
from PIL import Image

import torch
import torchvision

from utils import *


def collate_fn(batch):
    return tuple(zip(*batch))


class ObjDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, include_trig=False):
        self.root_dir = root_dir
        self.include_trig = include_trig

        # Load data
        self._load_data()

    def _load_data(self):
        # Load data from root_dir
        self.images = []
        self.targets = []
        self.triggers = []

        fns = [os.path.join(self.root_dir, fn) for fn in os.listdir(self.root_dir) if fn.endswith('.png')]
        for fn in fns:
            # Load the image
            self.images.append(fn)

            # Load the annotation
            json_file = fn.replace('.png', '.json')
            with open(json_file, 'r') as f:
                annotation = json.load(f)
            target = prepare_boxes(annotation, include_bg=True)
            self.targets.append(target)

            # Load the target object
            pt_file = fn.replace('.png', '_trigger.pt')
            if not os.path.exists(pt_file):
                self.triggers.append(None)
            else:
                trigger = torch.load(pt_file)
                self.triggers.append(trigger)

    def transform(self, fn):
        image = Image.open(fn)
        image = torchvision.transforms.ToTensor()(image)
        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        image = self.transform(image)
        target = self.targets[index]
        if self.include_trig:
            trigger = self.triggers[index]
            return image, target, trigger
        else:
            return image, target


class MixDataset(torch.utils.data.Dataset):
    def __init__(self, clean_dataset, poison_dataset, divide=10):
        self.clean_dataset = clean_dataset
        self.poison_dataset = poison_dataset
        self.divide = divide

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, index):
        # Split the clean and poison dataset by probability
        # 90% probability to get clean data
        # 10% probability to get poison data
        prob = index % self.divide
        if prob in [0]:
            # Randomly select a poison data
            rand_idx = np.random.randint(len(self.poison_dataset))
            return self.poison_dataset[rand_idx]
        else:
            # Randomly select a clean data
            rand_idx = np.random.randint(len(self.clean_dataset))
            return self.clean_dataset[rand_idx]
