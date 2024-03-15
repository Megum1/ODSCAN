import os
import sys
import cv2
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision
from torchvision.utils import save_image

import warnings

from models import *
from utils import *

warnings.filterwarnings("ignore")


def epsilon():
    return 1e-7


def mask_patn_process(mask, patn):
    mask_tanh = torch.tanh(mask) / (2 - epsilon()) + 0.5
    patn_tanh = torch.tanh(patn) / (2 - epsilon()) + 0.5
    return mask_tanh, patn_tanh


def nc(model_filepath, coco_folder):
    # load the model
    model = torch.load(model_filepath).cuda()
    model.eval()

    num_classes = 90

    model_id = model_filepath.split('/')[-2]
    save_path = f'result/{model_id}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ###########################################################
    # Pair selection
    ###########################################################
    # TODO: Can use Pre-processing to reduce the number of pairs
    null_class = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82]

    # Select all possible pairs for misclassification attack
    mis_pairs = []
    for victim_class in range(num_classes):
        if victim_class in null_class:
            continue
        for target_class in range(num_classes):
            if target_class == victim_class:
                continue
            mis_pairs.append((victim_class, target_class))

    # Select all possible pairs for evasion attack
    eva_pairs = []
    target_class = -1
    for victim_class in range(num_classes):
        if victim_class in null_class:
            continue
        eva_pairs.append((victim_class, target_class))

    # Merge the pairs
    all_pairs = mis_pairs + eva_pairs
    print(f'Number of pairs: {len(all_pairs)}')
    print(all_pairs)
    ###########################################################

    # Start inversion
    for victim_class, target_class in all_pairs:
        # Time for each pair
        time_start = time.time()

        # Load the coco dataset
        syn_images, syn_targets = prepare_dataset_coco(victim_class, coco_folder, n_per_class=10)
        syn_images, syn_targets = custom_preprocess(model, syn_images, syn_targets)

        # Image size
        h, w = syn_images.shape[2:]

        # Move to device
        with torch.no_grad():
            batch_images = syn_images.cuda()
            batch_targets = [{k: v.cuda() for k, v in t.items()} for t in syn_targets]

        # Initialization of parameters
        mask_init = np.random.random((1, 1, h, w))
        patn_init = np.random.random((1, 3, h, w))

        mask_init = np.arctanh((mask_init - 0.5) * (2 - epsilon()))
        patn_init = np.arctanh((patn_init - 0.5) * (2 - epsilon()))

        # Define optimizing parameters
        mask = torch.tensor(mask_init, dtype=torch.float, requires_grad=True, device=torch.device('cuda'))
        patn = torch.tensor(patn_init, dtype=torch.float, requires_grad=True, device=torch.device('cuda'))

        # Define the optimization
        optimizer = torch.optim.Adam(params=[mask, patn], lr=1e-1, betas=(0.5, 0.9))

        # Loss ce and weights
        criterion = torch.nn.CrossEntropyLoss()

        reg_best = 1 / epsilon()

        # Threshold for attack success rate
        init_asr_threshold = 0.9
        asr_threshold = init_asr_threshold

        # Initial cost for regularization
        init_cost = 1e-3
        cost = init_cost
        cost_multiplier_up = 2
        cost_multiplier_down = cost_multiplier_up ** 1.5

        # Counters for adjusting balance cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # Counter for early stop
        early_stop = True
        early_stop_threshold = 1.0
        early_stop_counter = 0
        early_stop_reg_best = reg_best

        # Patience
        patience = 5
        early_stop_patience = 5 * patience

        # Total optimization steps
        steps = 1000

        # Start optimization
        for step in range(steps):
            mask_tanh, patn_tanh = mask_patn_process(mask, patn)

            troj_images = patn_tanh * mask_tanh + batch_images * (1 - mask_tanh)

            # Model forward
            detections, cls_logits, resized_boxes = custom_forward(model, troj_images)

            # Box selection
            selected_cls_logits = []
            iou_threshold = 0.5
            mis_asr = 0
            eva_asr = 0
            for img_id in range(cls_logits.shape[0]):
                gt_boxes = batch_targets[img_id]['boxes']
                gt_labels = batch_targets[img_id]['labels']

                # Get the gt boxes with victim class
                gt_box = gt_boxes[gt_labels == victim_class]

                # Get the predicted boxes with target class
                pred_box = resized_boxes[img_id]
                if len(pred_box.shape) == 3:
                    pred_box = pred_box[:, target_class + 1, :]

                cur_iou = calc_iou(gt_box, pred_box).sum(dim=0)
                cur_cls_logits = cls_logits[img_id, cur_iou > iou_threshold, :]
                selected_cls_logits.append(cur_cls_logits)

                # TODO: Calculate the ASR
                # TODO: Check detection for mis and eva attack
                res_boxes = detections[img_id]['boxes']
                res_labels = detections[img_id]['labels']
                res_scores = detections[img_id]['scores']

                res_boxes, res_labels, res_scores = filter_low_scores(res_boxes, res_labels, res_scores, threshold=0.1)

                if len(res_boxes) == 0:
                    eva_asr += 1
                    continue

                res_iou = calc_iou(gt_box, res_boxes).sum(dim=0)
                sel_labels = res_labels[res_iou > iou_threshold]

                if (target_class + 1) in sel_labels:
                    mis_asr += 1
                if len(sel_labels) == 0:
                    eva_asr += 1

            mis_asr /= cls_logits.shape[0]
            eva_asr /= cls_logits.shape[0]

            selected_cls_logits = torch.cat(selected_cls_logits, dim=0)
            selected_targets = torch.ones(selected_cls_logits.shape[0], dtype=torch.long, device=torch.device('cuda')) * (target_class + 1)

            # Calculate ASR
            if target_class == -1:
                # Evasion attack
                asr = eva_asr
            else:
                # Misclassification attack
                asr = mis_asr

            ce_loss = criterion(selected_cls_logits, selected_targets)
            reg_loss = torch.abs(mask_tanh).sum()

            loss = ce_loss + cost * reg_loss

            eval_ce_loss = ce_loss.detach().cpu().item()
            eval_reg_loss = reg_loss.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                print(f'Pair: {victim_class}-{target_class}, Step: {step + 1}, cost: {cost:.3f}, CE_Loss: {eval_ce_loss:.3f}, Reg_Loss: {eval_reg_loss:.3f}, ASR: {asr:.2f}')

                flag_save = False
                if flag_save:
                    # Save the image
                    savefig = []
                    for i in range(len(detections)):
                        img = batch_images[i].detach().cpu()
                        img = (img * 255.).to(torch.uint8)
                        boxes = detections[i]['boxes'].detach().cpu()
                        labels = detections[i]['labels'].detach().cpu()
                        scores = detections[i]['scores'].detach().cpu()

                        new_boxes, new_labels, new_scores = filter_low_scores(boxes, labels, scores, threshold=0.1)

                        if len(new_boxes) == 0:
                            fig = img
                        else:
                            label_name = []
                            for id in new_labels:
                                # "-1" means removing the background
                                label_name.append(str(id.item() - 1))

                            fig = torchvision.utils.draw_bounding_boxes(
                                image=img, boxes=new_boxes, colors='red', labels=label_name, width=1, fill=False, font_size=200)

                        savefig.append(fig)

                    savefig = torch.stack(savefig, dim=0) / 255.0
                    save_image(savefig, f'{save_path}/visual_{victim_class}_{target_class}.png')

            if asr >= asr_threshold and eval_reg_loss < reg_best:
                # Update the best mask
                reg_best = eval_reg_loss
            
            # Check early stop
            if early_stop:
                # Only terminate if a valid attack has been found
                if reg_best < 1 / epsilon():
                    if reg_best >= early_stop_threshold * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and cost_up_flag and early_stop_counter >= early_stop_patience):
                    print('Early stop !\n')
                    break
            
            # Check cost modification
            if cost < epsilon() and asr >= asr_threshold:
                cost_set_counter += 1
                if cost_set_counter >= patience:
                    cost = init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('*** Initialize cost to %.2E' % (cost))
            else:
                cost_set_counter = 0
            
            if asr >= asr_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1
            
            if cost_up_counter >= patience:
                cost_up_counter = 0
                cost *= cost_multiplier_up
                cost_up_flag = True
                # print('UP cost to %.2E' % cost)
            if cost_down_counter >= patience:
                cost_down_counter = 0
                cost /= cost_multiplier_down
                cost_down_flag = True
                # print('DOWN cost to %.2E' % cost)
        
        # End time
        time_end = time.time()

        # Record results
        record_pair = f'{victim_class}_{target_class}'
        record_mask_size = reg_best
        record_asr = asr
        record_image_size = h * w
        record_time = time_end - time_start

        print(f'Results: {record_pair}, Mask size: {record_mask_size:.2f}, ASR: {record_asr}, Image Size: {record_image_size}, Time: {record_time:.2f}\n')

        # Save results
        np.save(f'{save_path}/nc_{record_pair}.npy', [record_mask_size, record_asr, record_image_size, record_time])


def pixel(model_filepath, coco_folder):
    # load the model
    model = torch.load(model_filepath).cuda()
    model.eval()

    num_classes = 90

    model_id = model_filepath.split('/')[-2]
    save_path = f'result/{model_id}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ###########################################################
    # Pair selection
    ###########################################################
    # TODO: Can use Pre-processing to reduce the number of pairs
    null_class = [11, 25, 28, 29, 44, 65, 67, 68, 70, 82]

    # Select all possible pairs for misclassification attack
    mis_pairs = []
    for victim_class in range(num_classes):
        if victim_class in null_class:
            continue
        for target_class in range(num_classes):
            if target_class == victim_class:
                continue
            mis_pairs.append((victim_class, target_class))

    # Select all possible pairs for evasion attack
    eva_pairs = []
    target_class = -1
    for victim_class in range(num_classes):
        if victim_class in null_class:
            continue
        eva_pairs.append((victim_class, target_class))

    # Merge the pairs
    all_pairs = mis_pairs + eva_pairs
    print(f'Number of pairs: {len(all_pairs)}')
    print(all_pairs)
    ###########################################################

    # Start inversion
    for victim_class, target_class in all_pairs:
        # Time for each pair
        time_start = time.time()

        # Load the coco dataset
        syn_images, syn_targets = prepare_dataset_coco(victim_class, coco_folder, n_per_class=10)
        syn_images, syn_targets = custom_preprocess(model, syn_images, syn_targets)

        # Image size
        h, w = syn_images.shape[2:]

        # Move to device
        with torch.no_grad():
            batch_images = syn_images.cuda()
            batch_targets = [{k: v.cuda() for k, v in t.items()} for t in syn_targets]

        # Define optimizing parameters
        pattern_shape = (1, 3, h, w)
        for i in range(2):
            init_pattern = np.random.random(pattern_shape)
            init_pattern = np.clip(init_pattern, 0.0, 1.0)

            if i == 0:
                pattern_pos_tensor = torch.Tensor(init_pattern).cuda()
                pattern_pos_tensor.requires_grad = True
            else:
                pattern_neg_tensor = torch.Tensor(- init_pattern).cuda()
                pattern_neg_tensor.requires_grad = True

        # Define the optimization
        optimizer = torch.optim.Adam(params=[pattern_pos_tensor, pattern_neg_tensor], lr=1e-1, betas=(0.5, 0.9))

        # Loss ce and weights
        criterion = torch.nn.CrossEntropyLoss()

        reg_best = 1 / epsilon()
        pixel_best = 1 / epsilon()

        # Threshold for attack success rate
        init_asr_threshold = 0.9
        asr_threshold = init_asr_threshold

        # Initial cost for regularization
        init_cost = 1e-3
        cost = init_cost
        cost_multiplier_up = 1.5
        cost_multiplier_down = cost_multiplier_up ** 1.5

        # Counters for adjusting balance cost
        cost_up_counter = 0
        cost_down_counter = 0

        # Patience
        patience = 10

        # Total optimization steps
        steps = 1000

        # Start optimization
        for step in range(steps):
            pattern_pos = torch.clamp(pattern_pos_tensor, 0.0, 1.0)
            pattern_neg = - torch.clamp(pattern_neg_tensor, 0.0, 1.0)

            troj_images = batch_images + pattern_pos + pattern_neg
            troj_images = torch.clamp(troj_images, 0.0, 1.0)

            # Model forward
            detections, cls_logits, resized_boxes = custom_forward(model, troj_images)

            # Box selection
            selected_cls_logits = []
            iou_threshold = 0.5
            mis_asr = 0
            eva_asr = 0
            for img_id in range(cls_logits.shape[0]):
                gt_boxes = batch_targets[img_id]['boxes']
                gt_labels = batch_targets[img_id]['labels']

                # Get the gt boxes with victim class
                gt_box = gt_boxes[gt_labels == victim_class]

                # Get the predicted boxes with target class
                pred_box = resized_boxes[img_id]
                if len(pred_box.shape) == 3:
                    pred_box = pred_box[:, target_class + 1, :]

                cur_iou = calc_iou(gt_box, pred_box).sum(dim=0)
                cur_cls_logits = cls_logits[img_id, cur_iou > iou_threshold, :]
                selected_cls_logits.append(cur_cls_logits)

                # TODO: Calculate the ASR
                # TODO: Check detection for mis and eva attack
                res_boxes = detections[img_id]['boxes']
                res_labels = detections[img_id]['labels']
                res_scores = detections[img_id]['scores']

                res_boxes, res_labels, res_scores = filter_low_scores(res_boxes, res_labels, res_scores, threshold=0.1)

                if len(res_boxes) == 0:
                    eva_asr += 1
                    continue

                res_iou = calc_iou(gt_box, res_boxes).sum(dim=0)
                sel_labels = res_labels[res_iou > iou_threshold]

                if (target_class + 1) in sel_labels:
                    mis_asr += 1
                if len(sel_labels) == 0:
                    eva_asr += 1

            mis_asr /= cls_logits.shape[0]
            eva_asr /= cls_logits.shape[0]

            selected_cls_logits = torch.cat(selected_cls_logits, dim=0)
            selected_targets = torch.ones(selected_cls_logits.shape[0], dtype=torch.long, device=torch.device('cuda')) * (target_class + 1)

            # Calculate ASR
            if target_class == -1:
                # Evasion attack
                asr = eva_asr
            else:
                # Misclassification attack
                asr = mis_asr

            ce_loss = criterion(selected_cls_logits, selected_targets)
            reg_pos  = torch.max(torch.tanh(pattern_pos_tensor / 10) / (2 - epsilon()) + 0.5, axis=0)[0]
            reg_neg  = torch.max(torch.tanh(pattern_neg_tensor / 10) / (2 - epsilon()) + 0.5, axis=0)[0]
            reg_loss = torch.sum(reg_pos) + torch.sum(reg_neg)

            loss = ce_loss + cost * reg_loss

            eval_ce_loss = ce_loss.detach().cpu().item()
            eval_reg_loss = reg_loss.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0:
                print(f'Pair: {victim_class}-{target_class}, Step: {step + 1}, cost: {cost:.3f}, CE_Loss: {eval_ce_loss:.3f}, Reg_Loss: {eval_reg_loss:.3f}, ASR: {asr:.2f}')

                flag_save = False
                if flag_save:
                    # Save the image
                    savefig = []
                    for i in range(len(detections)):
                        img = batch_images[i].detach().cpu()
                        img = (img * 255.).to(torch.uint8)
                        boxes = detections[i]['boxes'].detach().cpu()
                        labels = detections[i]['labels'].detach().cpu()
                        scores = detections[i]['scores'].detach().cpu()

                        new_boxes, new_labels, new_scores = filter_low_scores(boxes, labels, scores, threshold=0.1)

                        if len(new_boxes) == 0:
                            fig = img
                        else:
                            label_name = []
                            for id in new_labels:
                                # "-1" means removing the background
                                label_name.append(str(id.item() - 1))

                            fig = torchvision.utils.draw_bounding_boxes(
                                image=img, boxes=new_boxes, colors='red', labels=label_name, width=1, fill=False, font_size=200)

                        savefig.append(fig)

                    savefig = torch.stack(savefig, dim=0) / 255.0
                    save_image(savefig, f'{save_path}/visual_{victim_class}_{target_class}.png')

            # remove small pattern values
            threshold = 1.0 / 255.0
            pattern_pos_cur = pattern_pos.detach()
            pattern_neg_cur = pattern_neg.detach()
            pattern_pos_cur[(pattern_pos_cur < threshold) & (pattern_pos_cur > -threshold)] = 0
            pattern_neg_cur[(pattern_neg_cur < threshold) & (pattern_neg_cur > -threshold)] = 0
            pattern_cur = pattern_pos_cur + pattern_neg_cur

            # count current number of perturbed pixels
            pixel_cur = np.count_nonzero(np.sum(np.abs(pattern_cur.cpu().numpy()), axis=0))

            # Record the best pattern
            if asr >= asr_threshold and eval_reg_loss < reg_best and pixel_cur < pixel_best:
                pattern_pos_best = pattern_pos.detach()
                pattern_pos_best[pattern_pos_best < threshold] = 0
                init_pattern = pattern_pos_best
                with torch.no_grad():
                    pattern_pos_tensor.copy_(init_pattern)

                pattern_neg_best = pattern_neg.detach()
                pattern_neg_best[pattern_neg_best > -threshold] = 0
                init_pattern = - pattern_neg_best
                with torch.no_grad():
                    pattern_neg_tensor.copy_(init_pattern)

                pattern_best = pattern_pos_best + pattern_neg_best
                savefig = pattern_best.detach().cpu()

                reg_best = eval_reg_loss
                pixel_best = pixel_cur

                best_size = np.count_nonzero(pattern_best.abs().sum(0).cpu().numpy())
            
            # helper variables for adjusting loss weight
            if asr >= asr_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            # adjust loss weight
            if cost_up_counter >= patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = init_cost
                else:
                    cost *= cost_multiplier_up
            elif cost_down_counter >= patience:
                cost_down_counter = 0
                cost /= cost_multiplier_down
        
        if reg_best == 1 / epsilon():
            pattern_pos_best = pattern_pos.detach()
            pattern_pos_best[pattern_pos_best < threshold] = 0
            init_pattern = pattern_pos_best
            with torch.no_grad():
                pattern_pos_tensor.copy_(init_pattern)

            pattern_neg_best = pattern_neg.detach()
            pattern_neg_best[pattern_neg_best > -threshold] = 0
            init_pattern = - pattern_neg_best
            with torch.no_grad():
                pattern_neg_tensor.copy_(init_pattern)

            pattern_best = pattern_pos_best + pattern_neg_best
            savefig = pattern_best.detach().cpu()

            reg_best = eval_reg_loss
            pixel_best = pixel_cur

            best_size = np.count_nonzero(pattern_best.abs().sum(0).cpu().numpy())
        
        # End time
        time_end = time.time()

        # Record results
        record_pair = f'{victim_class}_{target_class}'
        record_mask_size = best_size
        record_asr = asr
        record_image_size = h * w
        record_time = time_end - time_start

        print(f'Results: {record_pair}, Mask size: {record_mask_size:.2f}, ASR: {record_asr}, Image Size: {record_image_size}, Time: {record_time:.2f}\n')

        # Save results
        np.save(f'{save_path}/pixel_{record_pair}.npy', [record_mask_size, record_asr, record_image_size, record_time])


if __name__ == "__main__":
    root = '/data/share/trojai/trojai-round10-v2-dataset'

    # Total number of models
    num_total = 128

    for i in range(num_total):
        # Your path of models trained on COCO dataset
        model_id = 'id-' + str(i).zfill(8)
        model_filepath = os.path.join(root, model_id, 'model.pt')
        # Your path of COCO dataset
        examples_dirpath = '/data3/share/large-example-data-all'

        nc(model_filepath, examples_dirpath)
        pixel(model_filepath, examples_dirpath)
