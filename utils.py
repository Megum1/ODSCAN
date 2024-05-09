import os
import sys
import time
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.ops import boxes as box_ops
from torchvision.utils import save_image

from collections import OrderedDict


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model(num_classes, network='ssd'):
    # Default: Load a pre-trained ssd300_vgg16 model
    # Object detection class ids start at 1, so total class count needs to match
    number_classes = num_classes + 1

    if network == 'ssd':
        # This loads the arch and a pre-trained backbone, but the object detector final classification layer is randomly initialized
        checkpoint = torchvision.models.detection.ssd300_vgg16(progress=True, trainable_backbone_layers=5, weights='DEFAULT').state_dict()
        # Remove the class weights, which need to match the number of classes
        del checkpoint["head.classification_head.module_list.0.weight"]
        del checkpoint["head.classification_head.module_list.0.bias"]
        del checkpoint["head.classification_head.module_list.1.weight"]
        del checkpoint["head.classification_head.module_list.1.bias"]
        del checkpoint["head.classification_head.module_list.2.weight"]
        del checkpoint["head.classification_head.module_list.2.bias"]
        del checkpoint["head.classification_head.module_list.3.weight"]
        del checkpoint["head.classification_head.module_list.3.bias"]
        del checkpoint["head.classification_head.module_list.4.weight"]
        del checkpoint["head.classification_head.module_list.4.bias"]
        del checkpoint["head.classification_head.module_list.5.weight"]
        del checkpoint["head.classification_head.module_list.5.bias"]

        model = torchvision.models.detection.ssd300_vgg16(progress=True, trainable_backbone_layers=5, num_classes=number_classes)
        # Overwrites the models wights (except for the final class layer) with the state dict values.
        # Strict false to allow missing keys (the class head)
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise Exception('Model structure not supported')

    return model


############################################
# Customized model forward for training
############################################
def ssd_custom_compute_loss(model, targets, head_outputs, anchors, matched_idxs):
    bbox_regression = head_outputs["bbox_regression"]
    cls_logits = head_outputs["cls_logits"]

    # Match original targets with default boxes
    num_foreground = 0
    bbox_loss = []
    cls_targets = []
    for (
        targets_per_image,
        bbox_regression_per_image,
        cls_logits_per_image,
        anchors_per_image,
        matched_idxs_per_image,
    ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
        # Produce the matching between boxes and targets
        foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
        foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
        num_foreground += foreground_matched_idxs_per_image.numel()

        # Calculate regression loss
        matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
        bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
        target_regression = model.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        bbox_loss.append(torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum"))

        # Estimate ground truth for class targets
        gt_classes_target = torch.zeros(
            (cls_logits_per_image.size(0),),
            dtype=targets_per_image["labels"].dtype,
            device=targets_per_image["labels"].device,
        )
        gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][foreground_matched_idxs_per_image]
        cls_targets.append(gt_classes_target)

    bbox_loss = torch.stack(bbox_loss)
    cls_targets = torch.stack(cls_targets)

    # Calculate classification loss
    num_classes = cls_logits.size(-1)

    # Calculate loss
    cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none").view(cls_targets.size())

    # Hard Negative Sampling
    foreground_idxs = cls_targets > 0
    num_negative = model.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
    negative_loss = cls_loss.clone()
    # Use -inf to detect positive values that creeped in the sample
    negative_loss[foreground_idxs] = -float("inf")
    values, idx = negative_loss.sort(1, descending=True)
    background_idxs = idx.sort(1)[1] < num_negative

    N = max(1, num_foreground)
    return {
        "bbox_regression": bbox_loss.sum() / N,
        "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
    }


def ssd_custom_forward(model, images, targets):
    # Get the original image sizes
    original_image_sizes = []
    for img in images:
        val = img.shape[-2:]
        original_image_sizes.append((val[0], val[1]))

    # Transform the input
    images, targets = model.transform(images, targets)

    # Get the features from the backbone
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    features = list(features.values())

    # Compute the ssd heads outputs using the features
    head_outputs = model.head(features)

    # Create the set of anchors
    anchors = model.anchor_generator(images, features)

    # Training phase: Compute the losses
    if model.training:
        losses = {}
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full(
                        (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                    )
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(model.proposal_matcher(match_quality_matrix))

        losses = ssd_custom_compute_loss(model, targets, head_outputs, anchors, matched_idxs)
        return losses

    # Evaluation phase: Post-process the detections
    else:
        detections = []
        detections = model.postprocess_detections(head_outputs, anchors, images.image_sizes)
        detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections


def custom_forward(model, images, targets):
    # Get model structure
    model_structure = model.__class__.__name__
    if model_structure == 'SSD':
        return ssd_custom_forward(model, images, targets)
    else:
        raise Exception('Model structure not supported')


############################################
# Customized model forward for scanning
############################################
def SSD_scan_forward(model, input_images, input_targets, get_logits=False):
    # Pre-process
    original_image_size = list(input_images[0].shape[-2:])
    images, targets = model.transform(input_images, input_targets)
    transformed_image_size = list(images.image_sizes[0])
    
    # Get logits w/o NMS
    image_shapes = images.image_sizes

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([('0', features)])
    features = list(features.values())

    # compute the ssd heads outputs using the features
    head_outputs = model.head(features)

    # create the set of anchors
    anchors = model.anchor_generator(images, features)
    detections = model.postprocess_detections(head_outputs, anchors, image_shapes)

    # Post-process
    batch_size = len(detections)
    image_sizes_list = [transformed_image_size for _ in range(batch_size)]
    original_image_sizes_list = [original_image_size for _ in range(batch_size)]
    detections = model.transform.postprocess(detections, image_sizes_list, original_image_sizes_list)

    # Get logits
    if get_logits:
        cls_logits = head_outputs['cls_logits']
        bbox_regression = head_outputs['bbox_regression']
        resized_boxes = []
        image_anchors = anchors
        
        for boxes, anchors, image_shape in zip(bbox_regression, image_anchors, image_shapes):
            boxes = model.box_coder.decode_single(boxes, anchors)
            boxes = torchvision.ops.boxes.clip_boxes_to_image(boxes, image_shape)
            resized_boxes.append(boxes)
        
        resized_boxes = torch.stack(resized_boxes)

        # Remember targets here shape is resized as (300, 300)
        cur_device = cls_logits.device

        ratios = [torch.tensor(s, dtype=torch.float32, device=cur_device) / torch.tensor(s_orig, dtype=torch.float32, device=cur_device)
                  for s, s_orig in zip(original_image_sizes_list, image_sizes_list)]
        ratios = torch.stack(ratios, dim=0)
        ratio_height, ratio_width = ratios.unbind(dim=1)
        ratios = torch.stack([ratio_width, ratio_height, ratio_width, ratio_height], dim=1)
        ratios = ratios[:, None, :]
        resized_boxes = resized_boxes * ratios

        return detections, cls_logits, resized_boxes
    
    # Get loss
    else:
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(model.proposal_matcher(match_quality_matrix))

        losses = model.compute_loss(targets, head_outputs, anchors, matched_idxs)

        loss_bbox = losses['bbox_regression']
        loss_cls = losses['classification']

        losses = [loss_bbox, loss_cls]

        return detections, losses


def scan_forward(model, images, targets, get_logits=False):
    # Get model structure
    model_structure = model.__class__.__name__
    if model_structure == 'SSD':
        return SSD_scan_forward(model, images, targets, get_logits)
    else:
        raise Exception('Model structure not supported')
############################################


def prepare_boxes(annotations, include_bg=False):
    if len(annotations) > 0:
        boxes = []
        class_ids = []
        for anno in annotations:
            bbox = [anno['bbox']['x1'], anno['bbox']['y1'], anno['bbox']['x2'], anno['bbox']['y2']]
            boxes.append(bbox)
            # The class id starts from 1 if background is included
            if include_bg:
                label = anno['label'] + 1
            else:
                label = anno['label']
            class_ids.append(label)

        class_ids = np.stack(class_ids)
        boxes = np.stack(boxes)
    else:
        class_ids = np.zeros((0))
        boxes = np.zeros((0, 4))

    target = {}
    target['boxes'] = torch.as_tensor(boxes)
    target['labels'] = torch.as_tensor(class_ids).type(torch.int64)
    return target


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
        new_boxes = torch.zeros((0, 4))
        new_labels = torch.zeros((0))
        new_scores = torch.zeros((0))
    else:
        new_boxes = torch.stack(new_boxes)
        new_labels = torch.stack(new_labels)
        new_scores = torch.stack(new_scores)
    return new_boxes, new_labels, new_scores


def util_save(raw_img, raw_anno, save_name):
    # Save image
    save_image(raw_img, f'{save_name}.png')

    # Save annotation
    boxes = raw_anno["boxes"]
    labels = raw_anno["labels"]
    annotations = []
    for _, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box

        # Convert to int
        x1 = x1.item()
        y1 = y1.item()
        x2 = x2.item()
        y2 = y2.item()
        label = label.item()

        anno = {}
        anno["bbox"] = {}
        anno["bbox"]["x1"] = x1
        anno["bbox"]["y1"] = y1
        anno["bbox"]["x2"] = x2
        anno["bbox"]["y2"] = y2
        anno["label"] = label
        annotations.append(anno)

    with open(f'{save_name}.json', 'w') as f:
        json.dump(annotations, f, indent=4)


# Sample a few examples for scanning
def prepare_dataset(class_id, examples_filepath, n_samples=32):
    images, targets = [], []
    for filename in os.listdir(examples_filepath):
        if len(images) >= n_samples:
            break
        if filename.endswith('.png'):
            img = np.array(Image.open(os.path.join(examples_filepath, filename))) / 255.
            img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))

            annos_file = filename.replace('.png', '.json')
            with open(os.path.join(examples_filepath, annos_file)) as f:
                annos = json.load(f)
            
            boxes, labels = [], []
            for anno in annos[1:]:
                bbox = anno['bbox']
                bbox = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
                label = anno['label']

                boxes.append(bbox)
                labels.append(label)
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            if class_id in labels:
                target = {"boxes": boxes, "labels": labels}
                targets.append(target)
                images.append(img)

    return images, targets


# Synthesize a few examples for scanning
def prepare_synth_dataset(label, foregrounds, backgrounds, n_per_class=32, min_fg_size=60, max_fg_size=80, bg_size=256, blur=True, center=False):
    # Images, targets
    images, targets = [], []

    for _ in range(n_per_class):
        if min_fg_size != max_fg_size:
            fg_size = np.random.randint(min_fg_size, max_fg_size)
        else:
            fg_size = min_fg_size

        fg_img = transforms.ToTensor()(Image.open(foregrounds[label]))
        bg_img = transforms.ToTensor()(Image.open(backgrounds[np.random.randint(0, len(backgrounds))]))

        # Resize foreground and background images
        fg_img = transforms.Resize((fg_size, fg_size))(fg_img)
        bg_img = transforms.Resize((bg_size, bg_size))(bg_img)

        # Foreground make and pattern
        mask, pattern = fg_img[3:, :, :], fg_img[:3, :, :]

        # Add Gaussian blur
        if blur:
            pattern = transforms.GaussianBlur(kernel_size=(3, 3), sigma=(9, 11))(pattern)

        # Foreground position
        if center:
            px, py = np.random.randint(fg_size, bg_size - 2 * fg_size, size=2)
        else:
            px, py = np.random.randint(0, bg_size - fg_size, size=2)

        r_mask = torch.zeros_like(bg_img)
        r_mask[:, px:px + fg_size, py:py + fg_size] = mask

        r_pattern = torch.zeros_like(bg_img)
        r_pattern[:, px:px + fg_size, py:py + fg_size] = pattern

        image = bg_img * (1 - r_mask) + r_pattern * r_mask
        bbox = [py, px, py + fg_size, px + fg_size]

        image = torchvision.transforms.functional.convert_image_dtype(image, torch.float)
        images.append(image)

        boxes = np.array([bbox])
        class_ids = np.array([label])

        target = {}
        target['boxes'] = torch.as_tensor(boxes)
        target['labels'] = torch.as_tensor(class_ids).type(torch.int64)
        targets.append(target)

    images = torch.stack(images)

    return images, targets


# Prepare background images
def prepare_background(backgrounds, n_samples=32, bg_size=256, blur=True):
    images = []

    for _ in range(n_samples):
        bg_img = transforms.ToTensor()(Image.open(backgrounds[np.random.randint(0, len(backgrounds))]))

        # Resize background images
        bg_img = transforms.Resize((bg_size, bg_size))(bg_img)

        # Add Gaussian blur
        if blur:
            bg_img = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(9, 11))(bg_img)

        image = torchvision.transforms.functional.convert_image_dtype(bg_img, torch.float)
        images.append(image)
    
    images = torch.stack(images)
    
    return images
