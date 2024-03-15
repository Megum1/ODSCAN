import os
import sys
import json
import random
import numpy as np
from PIL import Image, ImageDraw
from collections import OrderedDict

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.roi_heads import fastrcnn_loss

from transformers import *


def epsilon():
    return 1e-7


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def calc_iou(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A, B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A, B]
    union = area_a + area_b - inter
    return inter / union  # [A, B]


def prepare_dataset_coco(label, examples_filepath, n_per_class=10):
    # Load example images and annotations
    examples_dirpath = os.path.join(examples_filepath, f'class-{label + 1}')

    if not os.path.exists(examples_dirpath):
        # print(f'No examples found for class {label + 1}')
        return [], []

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.jpg')]
    fns = fns[:n_per_class]

    images, targets = [], []
    for fn in fns:
        image = np.array(Image.open(fn)) / 255.
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        images.append(image)

        # load the annotation
        with open(fn.replace('.jpg', '.json')) as json_file:
            # contains a list of coco annotation dicts
            annotations = json.load(json_file)
        
        if len(annotations) > 0:
            boxes = []
            class_ids = []
            for answer in annotations:
                boxes.append(answer['bbox'])
                class_ids.append(answer['category_id'] - 1)

            class_ids = np.stack(class_ids)
            boxes = np.stack(boxes)
            # convert [x, y, w, h] to [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        else:
            class_ids = np.zeros((0))
            boxes = np.zeros((0, 4))
        
        # remove boxes where the width or height is less than 8 pixels
        degenerate_boxes = (boxes[:, 2:] - boxes[:, :2]) < 8
        degenerate_boxes = np.sum(degenerate_boxes, axis=1)
        if degenerate_boxes.any():
            boxes = boxes[degenerate_boxes == 0, :]
            class_ids = class_ids[degenerate_boxes == 0]
        target = {}
        target['boxes'] = torch.as_tensor(boxes)
        target['labels'] = torch.as_tensor(class_ids).type(torch.int64)

        targets.append(target)

    return images, targets


def filter_low_scores(boxes, labels, scores, threshold=0.1):
    # Eliminate the trigger bbox if score is low
    new_boxes = []
    new_labels = []
    new_scores = []
    for i in range(len(boxes)):
        if scores[i] > threshold:
            new_boxes.append(boxes[i])
            new_labels.append(labels[i])
            new_scores.append(scores[i])
    
    if len(new_boxes) == 0:
        return [], [], []

    new_boxes = torch.stack(new_boxes)
    new_labels = torch.stack(new_labels)
    new_scores = torch.stack(new_scores)
    return new_boxes, new_labels, new_scores


############################################
# Customized model forward
############################################
def custom_preprocess(model, input_images, input_targets):
    # Pre-process
    images, targets = model.transform(input_images, input_targets)
    _, deprocess = get_norm(model)
    images = deprocess(images.tensors)
    return images, targets


def SSD_custom_forward(model, images):
    # Get logits w/o NMS
    image_shapes = images.image_sizes

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])
    features = list(features.values())

    # compute the ssd heads outputs using the features
    # the head outputs are the logits and the bounding box regression offsets
    head_outputs = model.head(features)

    # create the set of anchors
    # Initial anchors are the same for all images
    # Final anchors equal to the initial anchors + the bounding box regression offsets
    anchors = model.anchor_generator(images, features)
    detections = model.postprocess_detections(head_outputs, anchors, image_shapes)

    # Get logits
    cls_logits = head_outputs['cls_logits']
    bbox_regression = head_outputs['bbox_regression']
    resized_boxes = []
    image_anchors = anchors
    
    for boxes, anchors, image_shape in zip(bbox_regression, image_anchors, image_shapes):
        boxes = model.box_coder.decode_single(boxes, anchors)
        boxes = torchvision.ops.boxes.clip_boxes_to_image(boxes, image_shape)
        resized_boxes.append(boxes)
    
    # Remember targets here shape is resized as (300, 300)
    resized_boxes = torch.stack(resized_boxes)

    return detections, cls_logits, resized_boxes


def FasterRCNN_custom_forward(model, images):
    # Get logits w/o NMS
    image_shapes = images.image_sizes

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    
    # Get logits
    proposals, _ = model.rpn(images, features)
    detections, _ = model.roi_heads(features, proposals, image_shapes)

    # Get logits
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    cur_device = class_logits.device

    resized_boxes = []
    full_scores = []
    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = model.roi_heads.box_coder.decode(box_regression, proposals)
    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = class_logits.split(boxes_per_image, 0)
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = torchvision.ops.boxes.clip_boxes_to_image(boxes, image_shape)

        if boxes.shape[0] < 1000:
            boxes = torch.cat([boxes, torch.zeros((1000-boxes.shape[0], boxes.shape[1], boxes.shape[2])).to(cur_device)])
            scores = torch.cat([scores, torch.zeros((1000-scores.shape[0], scores.shape[1])).to(cur_device)])
        resized_boxes.append(boxes)
        full_scores.append(scores)
    
    num_classes = class_logits.shape[-1]

    # Remember targets here shape is resized as (800, 800)
    resized_boxes = torch.stack(resized_boxes).reshape((len(image_shapes), -1, num_classes, 4))
    cls_logits = torch.stack(full_scores).reshape((len(image_shapes), -1, num_classes))

    return detections, cls_logits, resized_boxes


def custom_forward(model, images):
    # Pre-transformation
    timages = pre_transform(model, images)

    # Get model structure
    model_structure = model.__class__.__name__
    if model_structure == 'SSD':
        return SSD_custom_forward(model, timages)
    elif model_structure == 'FasterRCNN':
        return FasterRCNN_custom_forward(model, timages)
    else:
        raise Exception('Model structure not supported')


def get_norm(model):
    ssd_mean = np.array([0.48235, 0.45882, 0.40784]).reshape((1,3,1,1))
    ssd_std = np.array([0.00392156862745098, 0.00392156862745098, 0.00392156862745098]).reshape((1,3,1,1))
    
    fasterrcnn_mean = np.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
    fasterrcnn_std = np.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))

    model_structure = model.__class__.__name__

    if model_structure == 'SSD':
        mean = torch.FloatTensor(ssd_mean).cuda()
        std = torch.FloatTensor(ssd_std).cuda()
    elif model_structure == 'FasterRCNN':
        mean = torch.FloatTensor(fasterrcnn_mean).cuda()
        std = torch.FloatTensor(fasterrcnn_std).cuda()
    else:
        raise NotImplementedError

    preprocess = transforms.Normalize(mean=mean, std=std)
    deprocess = transforms.Normalize(mean=-mean/std, std=1/std)
    return preprocess, deprocess


class ImageList:
    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


def images2imagelist(images):
    # images = list(image for image in images)
    image_sizes_list = []
    image_sizes = [img.shape[-2:] for img in images]
    for image_size in image_sizes:
        torch._assert(
            len(image_size) == 2,
            f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
        )
        image_sizes_list.append((image_size[0], image_size[1]))

    image_list = ImageList(images, image_sizes_list)
    return image_list


def pre_transform(model, images):
    preprocess, _ = get_norm(model)
    images = preprocess(images)
    images = images2imagelist(images)
    return images
