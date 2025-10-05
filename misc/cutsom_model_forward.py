import copy
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import fastrcnn_loss
from torchvision.models.detection import _utils as det_utils
from torchvision.ops import boxes as box_ops
from collections import OrderedDict
from transformers import DetrFeatureExtractor


############################################
# Customized model forward
############################################
def SSD_custom_forward(model, input_images, input_targets, get_logits=False):
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


def FasterRCNN_custom_forward(model, input_images, input_targets, get_logits=False):
    # Pre-process
    original_image_size = list(input_images[0].shape[-2:])
    images, targets = model.transform(input_images, input_targets)
    transformed_image_size = list(images.image_sizes[0])

    # Get logits w/o NMS
    image_shapes = images.image_sizes

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    
    # Get logits
    if get_logits:
        proposals, _ = model.rpn(images, features)
        detections, _ = model.roi_heads(features, proposals, image_shapes)

        # Post-process
        batch_size = len(detections)
        image_sizes_list = [transformed_image_size for _ in range(batch_size)]
        original_image_sizes_list = [original_image_size for _ in range(batch_size)]
        detections = model.transform.postprocess(detections, image_sizes_list, original_image_sizes_list)

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

        resized_boxes = torch.stack(resized_boxes).reshape((len(image_shapes), -1, num_classes, 4))
        cls_logits = torch.stack(full_scores).reshape((len(image_shapes), -1, num_classes))

        # Remember targets here shape is resized as (800, 800)
        ratios = [torch.tensor(s, dtype=torch.float32, device=cur_device) / torch.tensor(s_orig, dtype=torch.float32, device=cur_device)
                  for s, s_orig in zip(original_image_sizes_list, image_sizes_list)]
        ratios = torch.stack(ratios, dim=0)
        ratio_height, ratio_width = ratios.unbind(dim=1)
        ratios = torch.stack([ratio_width, ratio_height, ratio_width, ratio_height], dim=1)
        ratios = ratios[:, None, None, :]
        resized_boxes = resized_boxes * ratios
        return detections, cls_logits, resized_boxes

    # Get loss
    else:
        proposals, _ = model.rpn(images, features, targets)
        detections, _ = model.roi_heads(features, proposals, image_shapes, targets)

        # Post-process
        batch_size = len(detections)
        image_sizes_list = [transformed_image_size for _ in range(batch_size)]
        original_image_sizes_list = [original_image_size for _ in range(batch_size)]
        detections = model.transform.postprocess(detections, image_sizes_list, original_image_sizes_list)

        proposals, _, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(box_features)
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = [loss_box_reg, loss_classifier]

        return detections, losses


def Detr_custom_forward(model, input_images, input_targets, get_logits=False):
    # Pre-process
    images = torch.stack(input_images)
    h, w = images.shape[-2:]
    Detr_extractor = DetrFeatureExtractor()

    labels = []
    copy_targets = copy.deepcopy(input_targets)
    for target in copy_targets:
        label = {}
        label['class_labels'] = target['labels'].to(torch.long)

        raw_boxes = target['boxes'].to(torch.float32)
        # Convert to (0, 1)
        raw_boxes[:, 0] /= w
        raw_boxes[:, 1] /= h
        raw_boxes[:, 2] /= w
        raw_boxes[:, 3] /= h
        # Convert to (center_x, center_y, width, height)
        x_transposed = raw_boxes.T
        x0, y0, x1, y1 = x_transposed[0], x_transposed[1], x_transposed[2], x_transposed[3]
        b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
        label['boxes'] = torch.stack(b, axis=-1)

        labels.append(label)

    outputs = model(images, labels=labels)
    cls_logits = outputs['logits']
    loss_dict = outputs['loss_dict']

    # Post-process
    detr_sizes = [img.shape[-2:] for img in images]
    detr_sizes = torch.as_tensor(detr_sizes, dtype=torch.int32).to(images.device)
    detections = Detr_extractor.post_process_object_detection(outputs, target_sizes=detr_sizes, threshold=0.)

    # Get logits
    if get_logits:
        resized_boxes = []
        for i in range(len(detections)):
            resized_boxes.append(detections[i]['boxes'])
        resized_boxes = torch.stack(resized_boxes)
        
        return detections, cls_logits, resized_boxes

    # Get loss
    else:
        loss_classifier, loss_box_reg = loss_dict['loss_ce'], loss_dict['loss_bbox']
        losses = [loss_box_reg, loss_classifier]

        return detections, losses


def custom_forward(model, images, targets, get_logits=False):
    # Get model structure
    model_structure = model.__class__.__name__
    if model_structure == 'SSD':
        return SSD_custom_forward(model, images, targets, get_logits)
    elif model_structure == 'FasterRCNN':
        return FasterRCNN_custom_forward(model, images, targets, get_logits)
    elif model_structure == 'DetrForObjectDetection':
        return Detr_custom_forward(model, images, targets, get_logits)
    else:
        raise Exception('Model structure not supported')
