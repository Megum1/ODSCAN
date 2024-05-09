import os
import sys
import time
import numpy as np

import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image

from utils import *

import warnings
warnings.filterwarnings("ignore")


# Pre-processing to reduce the search space
def preprocessing(model_filepath, config, verbose=True):
    # Load some configurations
    seed = config['seed']
    examples_filepath = config['examples_filepath']
    num_classes = config['num_classes']
    batch_size = config['batch_size']
    iou_threshold = config['iou_threshold']
    topk = config['topk']

    # Set the random seed
    seed_torch(seed)

    # load the model
    model = torch.load(model_filepath).cuda()
    model.eval()

    # Record logits
    all_logits = {}

    if verbose:
        print('=' * 30, 'Preprocessing to reduce search space', '=' * 30)

    for victim_class in range(num_classes):
        if verbose:
            print(f'Processing victim class {victim_class}/{num_classes - 1}')

        # Prepare the dataset
        images, targets = prepare_dataset(victim_class, examples_filepath)

        num_batches = int(np.ceil(len(images) / batch_size))

        # Batch processing
        cls_logits, resized_boxes = [], []
        for batch_idx in range(num_batches):
            # Get the batch
            batch_images = images[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_targets = targets[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            # Move to device
            batch_images = [img.cuda() for img in batch_images]
            batch_targets = [{k: v.cuda() for k, v in t.items()} for t in batch_targets]
            
            _, batch_cls_logits, batch_resized_boxes = scan_forward(model, batch_images, batch_targets, get_logits=True)

            cls_logits.append(batch_cls_logits.detach().cpu())
            resized_boxes.append(batch_resized_boxes.detach().cpu())
        
        cls_logits = torch.cat(cls_logits, dim=0)
        resized_boxes = torch.cat(resized_boxes, dim=0)

        # Check the logits
        for i in range(len(images)):
            cls_logit = cls_logits[i]
            resized_box = resized_boxes[i]

            fps = F.softmax(cls_logit, dim=-1)
            # gt_box.shape: torch.Size([1, 4])
            gt_box = targets[i]['boxes']

            # There may be multiple gt_boxes
            for gt_box_id in range(gt_box.shape[0]):
                # Only care the victim class
                gt_label = targets[i]['labels'][gt_box_id].item()
                if gt_label != victim_class:
                    continue

                # Take the current gt_box
                cur_box = gt_box[gt_box_id].unsqueeze(0)

                if len(resized_box.shape) == 3:
                    iou_output = []
                    for box_id in range(resized_box.shape[0]):
                        box = resized_box[box_id].view(-1, 4)
                        iou_output.append(calc_iou(box, cur_box))
                    iou_output = torch.stack(iou_output, dim=0)
                    iou_output = iou_output.view(resized_box.shape[0], -1)
                else:
                    box = resized_box.view(-1, 4)
                    iou_output = calc_iou(box, cur_box)[:, 0]

                # Traverse all the possible target classes
                for target_class in range(num_classes):
                    # Skip the victim class
                    if target_class == victim_class:
                        continue

                    # Only care the victim class (remember to remove the background class)
                    if len(resized_box.shape) == 3:
                        cur_iou_output = iou_output[:, target_class + 1]
                    else:
                        cur_iou_output = iou_output

                    box_mask = (cur_iou_output > iou_threshold).int()

                    # Take the index in box_mask with non-zero value
                    nonzero_idx = torch.nonzero(box_mask).view(-1).tolist()

                    # Get the logits of valid masks
                    logits = 0
                    if len(nonzero_idx) > 0:
                        logits = fps[nonzero_idx]
                        # Remove the background class
                        logits = logits[:, 1:]
                        # Remove non-object class
                        if logits.shape[-1] == num_classes + 1:
                            logits = logits[:, :-1]

                        # Get the max logits of the victim class
                        # get the prediction of logits
                        logits = logits[:, target_class].max().item()

                    pair_id = f'{victim_class}_{target_class}'
                    if pair_id not in all_logits:
                        all_logits[pair_id] = []
                    all_logits[pair_id].append(logits)

    # Get the mean logits for each pair
    for pair_id in all_logits:
        all_logits[pair_id] = np.mean(all_logits[pair_id])

    post_process = ['max', 'max_diff'][1]
    # Take the max logits
    if post_process == 'max':
        results = all_logits
    # Take the max logits difference
    elif post_process == 'max_diff':
        results = {}
        for victim in range(num_classes):
            for target in range(num_classes):
                if victim == target:
                    continue
                pair_id = f'{victim}_{target}'
                reverse_pair_id = f'{target}_{victim}'
                diff = all_logits[pair_id] - all_logits[reverse_pair_id]
                if diff > 0:
                    results[pair_id] = diff
    else:
        raise Exception('Post process not supported')

    # Select the top pair
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_results)
    top_pairs = sorted_results[:topk]
    return top_pairs


############################################
# Trigger inversion
############################################
def place_bg_trigger(obj_boxes, img_len, trig_len=32):
    # There may be multiple objects in the image
    boxes = []
    for obj_bbox in obj_boxes:
        py1, px1, py2, px2 = obj_bbox
        boxes.append([px1, py1, px2, py2])

    # Stamp the trigger at background
    while True:
        rx, ry = np.random.randint(0, img_len - trig_len, 2)

        overlap = False
        for box in boxes:
            px1, py1, px2, py2 = box
            if (px1 - trig_len <= ry <= px2 and py1 - trig_len <= rx <= py2):
                overlap = True
                break

        if not overlap:
            break

    return rx, ry, trig_len


def place_fg_trigger(gt_bbox, size_rate=0.1, over=2):
    x1, y1, x2, y2 = gt_bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Resize the trigger
    # box_len = min(x2 - x1, y2 - y1)
    box_len = int((x2 - x1 + y2 - y1) / 2)
    trig_len = int(box_len * size_rate)

    # Stamp trigger at the object
    x_bias = np.random.randint(-trig_len // over, trig_len // over)
    y_bias = np.random.randint(-trig_len // over, trig_len // over)

    x1 = (x1 + x2) // 2 - trig_len // 2 + x_bias
    y1 = (y1 + y2) // 2 - trig_len // 2 + y_bias

    return x1, y1, trig_len
############################################


def comp_zero(diff, temperature):
    return torch.tanh(diff * temperature) / 2 + 0.5


def polygon_trigger_inversion(model_filepath, victim_class, target_class, location, config, verbose=True):
    # Load some configurations
    seed = config['seed']
    foreground_folder = config['foreground_folder']
    background_folder = config['background_folder']
    batch_size = config['batch_size']
    n_samples = config['n_samples']
    save_folder = config['save_folder']
    iou_threshold = config['iou_threshold']
    conf_threshold = config['conf_threshold']
    epochs = config['epochs']
    warmup_epochs = int(epochs * 0.7)

    # Set the random seed
    seed_torch(seed)

    # load the model
    model = torch.load(model_filepath).cuda()
    model.eval()

    # Prepare the dataset
    foregrounds = [os.path.join(foreground_folder, f) for f in os.listdir(foreground_folder)]
    foregrounds = np.sort(foregrounds)
    backgrounds = [os.path.join(background_folder, f) for f in os.listdir(background_folder)]

    syn_images, syn_targets = prepare_synth_dataset(victim_class, foregrounds, backgrounds, n_per_class=n_samples)
    img_len = syn_images[0].shape[1]

    # Generate trigger position and size according to the objects
    trig_info = []
    for i in range(len(syn_images)):
        image = syn_images[i]
        target = syn_targets[i]

        obj_boxes = target['boxes']
        obj_labels = target['labels']

        # Stamp trigger on "one" victim object
        for obj_id in range(len(obj_boxes)):
            cur_bbox = obj_boxes[obj_id]
            cur_label = obj_labels[obj_id]
            if cur_label == victim_class:
                gt_bbox = cur_bbox
                break

        # Get trigger position and size
        if location == 'background':
            x1, y1, trig_len = place_bg_trigger(obj_boxes, img_len)
        elif location == 'foreground':
            x1, y1, trig_len = place_fg_trigger(gt_bbox)
        else:
            raise Exception('Location not supported')

        trig_info.append((x1, y1, trig_len, gt_bbox))

    # Special polygon trigger generation function
    # Randomly sample 4 floating points between [0, 1]
    dist_init = np.array([0.2, 0.8, 0.8, 0.2])
    patn_init = np.random.random((3, 1, 1))

    # Define optimizing parameters
    optm_dist = torch.tensor(dist_init, dtype=torch.float, requires_grad=True, device=torch.device('cuda'))
    optm_patn = torch.tensor(patn_init, dtype=torch.float, requires_grad=True, device=torch.device('cuda'))

    optimizer = torch.optim.Adam(params=[optm_dist, optm_patn], lr=1e-1, betas=(0.5, 0.9))
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if verbose:
        print('=' * 30, f'Scan object misclassification attack: victim-{victim_class}, target-{target_class}', '=' * 30)

    # Start the trigger inversion
    for epoch in range(epochs):
        # Batch processing
        num_batches = int(np.ceil(len(syn_images) / batch_size))

        # Record the decision metrics
        eval_loss = 0
        eval_boxes = []
        eval_labels = []
        eval_scores = []
        raw_images = []

        # Decision metrics
        asr = 0
        target_conf = []

        # Randomly shuffle the data
        idx = np.arange(len(syn_images))
        np.random.shuffle(idx)
        syn_images = [syn_images[i] for i in idx]
        syn_targets = [syn_targets[i] for i in idx]
        trig_info = [trig_info[i] for i in idx]

        for batch_idx in range(num_batches):
            # Get the batch
            batch_images = syn_images[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_targets = syn_targets[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_trig_info = trig_info[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            # Reset gradients
            optimizer.zero_grad()

            # Stamp the trigger on the image
            troj_images = []
            troj_targets = []

            # Trigger pattern
            patn_clip = torch.clamp(optm_patn, 0, 1)

            # Trigger mask
            for i in range(len(batch_images)):
                cur_image = batch_images[i].cuda()
                x_loc, y_loc, trig_len, gt_bbox = batch_trig_info[i]
                
                cur_target = {}
                cur_target['boxes'] = gt_bbox.view(1, 4).cuda()
                cur_target['labels'] = torch.as_tensor([target_class]).type(torch.int64).cuda()

                # Clip the optimaization distance
                dist_clip = torch.clamp(optm_dist, 0, 1) * (trig_len - 1)

                # Define the theta mask
                range_mask = torch.arange(trig_len)
                def_x_mask = range_mask.unsqueeze(0).repeat(trig_len, 1).cuda()
                def_y_mask = range_mask.unsqueeze(1).repeat(1, trig_len).cuda()

                # Operation on the dist mask
                dist_mask = torch.ones((1, trig_len, trig_len)).cuda()

                coef_x_list = [dist_clip[1], - dist_clip[3], trig_len - 1 - dist_clip[1], dist_clip[3] - trig_len + 1]
                coef_y_list = [dist_clip[0], trig_len - 1 - dist_clip[0], - dist_clip[2], dist_clip[2] - trig_len + 1]
                bias_list = [- dist_clip[0] * dist_clip[1], dist_clip[0] * dist_clip[3], dist_clip[1] * dist_clip[2], (trig_len - 1) * (trig_len - 1) - dist_clip[2] * dist_clip[3]]

                for i in range(4):
                    coef_x = coef_x_list[i]
                    coef_y = coef_y_list[i]
                    bias = bias_list[i]
                    
                    cur_mask = coef_x * def_x_mask + coef_y * def_y_mask + bias
                    cur_mask = comp_zero(cur_mask, temperature=10)
                    dist_mask = dist_mask * cur_mask
                
                # Padding the mask to the image size
                cur_mask = F.pad(dist_mask, (x_loc, img_len - trig_len - x_loc, y_loc, img_len - trig_len - y_loc), 'constant', 0)
                cur_patn = patn_clip

                troj_image = cur_image * (1 - cur_mask) + cur_patn * cur_mask
                troj_images.append(troj_image)
                troj_targets.append(cur_target)

            batch_detections, batch_cls_logits, batch_resized_boxes = scan_forward(model, troj_images, troj_targets, get_logits=True)

            select_cls_logits = []
            # Select foreground boxes
            for i in range(len(troj_images)):
                cur_boxes = batch_detections[i]['boxes'].detach().cpu()
                cur_labels = batch_detections[i]['labels'].detach().cpu()
                cur_scores = batch_detections[i]['scores'].detach().cpu()

                eval_boxes.append(cur_boxes)
                eval_labels.append(cur_labels)
                eval_scores.append(cur_scores)
                raw_images.append(troj_images[i].detach().cpu())

                # Filter low score boxes with threshold
                new_boxes, new_labels, new_scores = filter_low_scores(cur_boxes, cur_labels, cur_scores, threshold=conf_threshold)

                # gt_bbox.shape: torch.Size([1, 4])
                x_loc, y_loc, trig_len, gt_bbox = batch_trig_info[i]
                gt_bbox = gt_bbox.unsqueeze(0)
                trig_box = [x_loc, y_loc, x_loc + trig_len, y_loc + trig_len]
                trig_box = torch.as_tensor(trig_box, dtype=torch.float32).view(1, 4)

                flag = 0
                for j in range(len(new_boxes)):
                    cur_bbox = new_boxes[j].unsqueeze(0)
                    cur_label = new_labels[j]
                    cur_score = new_scores[j]
                    # Check foreground boxes
                    if calc_iou(cur_bbox, gt_bbox).sum() > iou_threshold:
                        # Correctly misclassified
                        if cur_label == target_class + 1:
                            flag = 1

                # Record the ASR
                asr += flag

                # Only one gt_bbox: torch.Size([1, 4])
                gt_bbox = gt_bbox.cuda()

                cls_logit = batch_cls_logits[i]
                resized_box = batch_resized_boxes[i]

                if len(resized_box.shape) == 3:
                    iou_output = []
                    for box_id in range(resized_box.shape[0]):
                        box = resized_box[box_id].view(-1, 4)
                        iou_output.append(calc_iou(box, gt_bbox))
                    iou_output = torch.stack(iou_output, dim=0)
                    iou_output = iou_output.view(resized_box.shape[0], -1)
                    iou_output = iou_output[:, target_class + 1]
                else:
                    box = resized_box.view(-1, 4)
                    iou_output = calc_iou(box, gt_bbox)[:, 0]

                box_mask = (iou_output > iou_threshold)

                cur_cls = cls_logit[box_mask]

                if len(cur_cls) == 0:
                    continue
                select_cls_logits.append(cur_cls)

                tar_logit = torch.nn.functional.softmax(cur_cls, dim=1)[: , target_class + 1].max().item()
                target_conf.append(tar_logit)

            # Remove logits if there is no foreground box
            if len(select_cls_logits) == 0:
                continue

            # Concat the logits
            select_cls_logits = torch.cat(select_cls_logits, dim=0)
            target_cls_logits = torch.LongTensor([target_class + 1]).cuda().repeat(select_cls_logits.shape[0])

            # Compute the loss
            if epoch > warmup_epochs:
                # Add weights to each box
                weights = torch.nn.functional.softmax(select_cls_logits, dim=0)[:, target_class + 1]
                loss = criterion(select_cls_logits, target_cls_logits)
                loss = torch.mean(loss * weights)
            else:
                loss = criterion(select_cls_logits, target_cls_logits).mean()

            eval_loss += loss.item()

            # Backward
            loss.backward()
            optimizer.step()

        # Compute the decision metrics
        eval_loss = eval_loss / num_batches
        asr = asr / len(syn_images)
        target_conf = np.mean(target_conf)

        if verbose:
            # Save the image
            savefig = []
            for i in range(len(eval_boxes)):
                img = (raw_images[i] * 255.).to(torch.uint8)
                boxes = eval_boxes[i]
                labels = eval_labels[i]
                scores = eval_scores[i]

                new_boxes, new_labels, new_scores = filter_low_scores(boxes, labels, scores, threshold=conf_threshold)

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
            
            # Define the saving path
            model_id = model_filepath.split('/')[-1].split('.')[-2]
            save_path = os.path.join(save_folder, model_id)
            # Create a new directory if it does not exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = os.path.join(save_path, f'{location}_{victim_class}_{target_class}.png')            
            savefig = torch.stack(savefig, dim=0) / 255.0
            save_image(savefig, save_name, dpi=600)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {eval_loss:.4f}, ASR: {asr*100:.2f}%, Target-confidence: {target_conf:.4f}')

    return asr, target_conf


# Main function
def main(config, verbose=True):
    # The subject model under scan
    model_filepath = config['model_filepath']

    # Step 1: Pre-processing to reduce the search space
    potential_pairs = preprocessing(model_filepath, config, verbose=verbose)
    print('=' * 50)
    print(f'Potential pairs: {potential_pairs}')

    # Step 2 : Scan the potential pairs for background triggers
    best_victim, best_target, best_asr, best_conf = 0, 0, 0, 0
    location = 'background'
    for rank in range(len(potential_pairs)):
        potential_pair = potential_pairs[rank]
        victim_class, target_class = potential_pair[0].split('_')
        victim_class = int(victim_class)
        target_class = int(target_class)

        # Trigger inversion
        asr, conf = polygon_trigger_inversion(model_filepath, victim_class, target_class, location, config, verbose=verbose)
        if asr + conf > best_asr + best_conf:
            best_victim, best_target, best_asr, best_conf = victim_class, target_class, asr, conf
    
    print('=' * 50)
    print(f'[Background] Most malicious class: {best_victim}-{best_target}, ASR: {best_asr*100:.2f}%, Target-confidence: {best_conf:.4f}')
    background_result = {'victim': best_victim, 'target': best_target, 'asr': best_asr, 'conf': best_conf}

    # Step 3 : Scan the potential pairs for foreground triggers
    best_victim, best_target, best_asr, best_conf = 0, 0, 0, 0
    location = 'foreground'
    for rank in range(len(potential_pairs)):
        potential_pair = potential_pairs[rank]
        victim_class, target_class = potential_pair[0].split('_')
        victim_class = int(victim_class)
        target_class = int(target_class)

        # Trigger inversion
        asr, conf = polygon_trigger_inversion(model_filepath, victim_class, target_class, location, config, verbose=verbose)
        if asr + conf > best_asr + best_conf:
            best_victim, best_target, best_asr, best_conf = victim_class, target_class, asr, conf
    
    print('=' * 50)
    print(f'[Foreground] Most malicious class: {best_victim}-{best_target}, ASR: {best_asr*100:.2f}%, Target-confidence: {best_conf:.4f}')
    foreground_result = {'victim': best_victim, 'target': best_target, 'asr': best_asr, 'conf': best_conf}

    return background_result, foreground_result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Scan object misclassification attack')
    parser.add_argument('--seed', default=1024, type=int, help='random seed')
    parser.add_argument('--model_filepath', default='ckpt/ssd_poison_misclassification_foreground_0_3.pt', help='model filepath')
    parser.add_argument('--num_classes', default=5, type=int, help='number of classes')
    parser.add_argument('--examples_filepath', default='data/test', type=str, help='examples filepath')
    parser.add_argument('--foreground_folder', default='data/foregrounds', type=str, help='foreground folder')
    parser.add_argument('--background_folder', default='data/backgrounds', type=str, help='background folder')
    parser.add_argument('--n_samples', default=5, type=int, help='number of background samples')
    parser.add_argument('--trig_len', default=32, type=int, help='trigger length')
    parser.add_argument('--trig_bias', default=10, type=int, help='trigger bias')
    parser.add_argument('--save_folder', default='invert_misclassification', type=str, help='save folder')
    parser.add_argument('--iou_threshold', default=0.5, type=float, help='iou threshold')
    parser.add_argument('--conf_threshold', default=0.05, type=float, help='confidence threshold')
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs')
    parser.add_argument('--topk', default=3, type=int, help='topk pairs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--verbose', default=1, type=int, help='verbose')

    args = parser.parse_args()

    # Scan the subject model
    config = args.__dict__
    verbose = True if config['verbose'] == 1 else False
    bg_result, fg_result = main(config, verbose=verbose)

    # Decide the final result
    print('\n\n')
    if bg_result['asr'] > 0.9 and bg_result['conf'] > 0.2:
        print(f'[Decision] The model is vulnerable to the [background] object misclassification attack')
    elif fg_result['asr'] > 0.9 and fg_result['conf'] > 0.8:
        print(f'[Decision] The model is vulnerable to the [foreground] object misclassification attack')
    else:
        print(f'[Decision] The model is clean against the object misclassification attack')
