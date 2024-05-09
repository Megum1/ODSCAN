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


# Transform non-differentiable comparison functions into differentiable
def comp_zero(diff, temperature):
    return torch.tanh(diff * temperature) / 2 + 0.5


# Polygon-region trigger inversion
def polygon_trigger_inversion(model_filepath, victim_class, config, verbose=True):
    # Load some configurations
    seed = config['seed']
    background_folder = config['background_folder']
    n_samples = config['n_samples']
    save_folder = config['save_folder']
    iou_threshold = config['iou_threshold']
    conf_threshold = config['conf_threshold']
    epochs = config['epochs']
    warmup_epochs = int(epochs * 0.7)
    # Trigger configurations
    trig_len = config['trig_len']
    trig_bias = config['trig_bias']

    # Set the random seed
    seed_torch(seed)

    # Load the model
    model = torch.load(model_filepath).cuda()
    model.eval()

    # Sample a few background images
    backgrounds = [os.path.join(background_folder, f) for f in os.listdir(background_folder)]
    syn_images = prepare_background(backgrounds, n_samples, blur=True)

    # Initialization
    img_len = syn_images[0].shape[-1]
    trig_bbox = torch.tensor([[trig_bias, trig_bias, trig_bias + trig_len, trig_bias + trig_len]])

    # Target class is equal to the victim class
    # Specially for the object appearing attack
    target_class = victim_class

    # Define the polygon mask
    range_mask = torch.arange(trig_len)
    def_x_mask = range_mask.unsqueeze(0).repeat(trig_len, 1).cuda()
    def_y_mask = range_mask.unsqueeze(1).repeat(1, trig_len).cuda()

    # Initialize the trigger mask and pattern
    dist_init = np.random.random((8,)) * 0.01
    dist_init[-2:] = 0.9 - dist_init[:2]
    patn_init = np.random.random((3, 1, 1))

    # Define optimizing parameters
    optm_dist = torch.tensor(dist_init, dtype=torch.float, requires_grad=True, device=torch.device('cuda'))
    optm_patn = torch.tensor(patn_init, dtype=torch.float, requires_grad=True, device=torch.device('cuda'))

    optimizer = torch.optim.Adam(params=[optm_dist, optm_patn], lr=1e-1, betas=(0.5, 0.9))
    criterion = torch.nn.CrossEntropyLoss()

    if verbose:
        print('=' * 30, f'Scan object appearing attack: label-{victim_class}', '=' * 30)

    for epoch in range(epochs):
        batch_images = syn_images.cuda()
        batch_targets = []
        for _ in range(len(batch_images)):
            trig_target = {}
            trig_target['boxes'] = trig_bbox
            trig_target['labels'] = torch.as_tensor([victim_class + 1]).reshape((1, ))
            batch_targets.append(trig_target)

        # Reset gradients
        optimizer.zero_grad()

        # Clip the parameters
        dist_clip = torch.clamp(optm_dist, 0, 1) * trig_len
        patn_clip = torch.clamp(optm_patn, 0, 1)

        # Operation on the dist mask
        dist_mask = torch.ones((1, trig_len, trig_len)).cuda()
        u = trig_len - 1
        coef_x_list = [dist_clip[4], - dist_clip[6], u - dist_clip[5], dist_clip[7] - u]
        coef_y_list = [dist_clip[0], - dist_clip[1] + u, - dist_clip[2], dist_clip[3] - u]
        bias_list = [- dist_clip[0] * dist_clip[4], dist_clip[1] * dist_clip[6], dist_clip[2] * dist_clip[5], - dist_clip[3] * dist_clip[7] + u * u]

        for i in range(4):
            coef_x = coef_x_list[i]
            coef_y = coef_y_list[i]
            bias = bias_list[i]
            
            cur_mask = coef_x * def_x_mask + coef_y * def_y_mask + bias
            cur_mask = comp_zero(cur_mask, temperature=10)
            dist_mask = dist_mask * cur_mask

        # Padding the theta mask to (256, 256) with 0
        trigger_mask = F.pad(dist_mask, (trig_bias, img_len - trig_len - trig_bias, trig_bias, img_len - trig_len - trig_bias), 'constant', 0)

        if patn_clip.shape[-1] == 1:
            trigger_pattern = patn_clip
        else:
            trigger_pattern = F.pad(patn_clip, (trig_bias, img_len - trig_len - trig_bias, trig_bias, img_len - trig_len - trig_bias), 'constant', 0)
        trig_images = batch_images * (1 - trigger_mask) + trigger_pattern * trigger_mask

        raw_images = trig_images.clone()

        # Split the batch into a list of images to fit the model
        trig_images = list(image for image in trig_images)
        trig_targets = [{k: v.cuda() for k, v in t.items()} for t in batch_targets]

        # Compute the loss
        batch_detections, batch_cls_logits, batch_resized_boxes = scan_forward(model, trig_images, trig_targets, get_logits=True)

        # Record the metrics
        asr = 0
        target_conf = []
        select_cls_logits = []

        # Select foreground boxes
        for i in range(len(trig_images)):
            cur_boxes = batch_detections[i]['boxes'].detach().cpu()
            cur_labels = batch_detections[i]['labels'].detach().cpu()
            cur_scores = batch_detections[i]['scores'].detach().cpu()

            # Filter low score boxes according to the threshold
            new_boxes, new_labels, new_scores = filter_low_scores(cur_boxes, cur_labels, cur_scores, threshold=conf_threshold)

            # gt_bbox.shape: torch.Size([1, 4])
            gt_bbox = trig_bbox

            # Check if the trigger is effective
            flag = 0
            for j in range(len(new_boxes)):
                cur_bbox = new_boxes[j].unsqueeze(0)
                cur_label = new_labels[j]
                cur_score = new_scores[j]
                if calc_iou(cur_bbox, gt_bbox).sum() > iou_threshold:
                    # Correctly misclassified
                    if cur_label == target_class + 1:
                        flag = 1

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

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Compute the decision metrics
        eval_loss = loss.item()
        asr = asr / len(trig_images)
        target_conf = np.mean(target_conf)

        if verbose:
            # Save the image
            savefig = []
            for i in range(len(batch_detections)):
                img = raw_images[i].detach().cpu()
                img = (img * 255.).to(torch.uint8)
                boxes = batch_detections[i]['boxes'].detach().cpu()
                labels = batch_detections[i]['labels'].detach().cpu()
                scores = batch_detections[i]['scores'].detach().cpu()

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
            
            # Define the saving path
            model_id = model_filepath.split('/')[-1].split('.')[-2]
            save_path = os.path.join(save_folder, model_id)
            # Create a new directory if it does not exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_name = os.path.join(save_path, f'label_{target_class}.png')
            savefig = torch.stack(savefig, dim=0) / 255.0
            save_image(savefig, save_name, dpi=600)

            print(f'Epoch {epoch+1}/{epochs}, Loss: {eval_loss:.4f}, ASR: {asr*100:.2f}%, Target-confidence: {target_conf:.4f}')

    return asr, target_conf


# Main function
def main(config, verbose=True):
    # Scan for all labels since there are not many classes
    # One can add a warmup stage here to reduce the search space
    model_filepath = config['model_filepath']

    best_class, best_asr, best_conf = 0, 0, 0
    for victim_class in range(config['num_classes']):
        asr, conf = polygon_trigger_inversion(model_filepath, victim_class=victim_class, config=config, verbose=verbose)
        if asr + conf > best_asr + best_conf:
            best_class = victim_class
            best_asr = asr
            best_conf = conf

    print('=' * 50)
    print(f'Most malicious class: {best_class}, ASR: {best_asr*100:.2f}%, Target-confidence: {best_conf:.4f}')

    result = {'target': best_class, 'asr': best_asr, 'conf': best_conf}
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Scan object appearing attack')
    parser.add_argument('--seed', default=1024, type=int, help='random seed')
    parser.add_argument('--model_filepath', default='ckpt/ssd_poison_appearing_background_0_3.pt', help='model filepath')
    parser.add_argument('--num_classes', default=5, type=int, help='number of classes')
    parser.add_argument('--background_folder', default='data/backgrounds', type=str, help='background folder')
    parser.add_argument('--n_samples', default=5, type=int, help='number of background samples')
    parser.add_argument('--trig_len', default=32, type=int, help='trigger length')
    parser.add_argument('--trig_bias', default=10, type=int, help='trigger bias')
    parser.add_argument('--save_folder', default='invert_appearing', type=str, help='save folder')
    parser.add_argument('--iou_threshold', default=0.5, type=float, help='iou threshold')
    parser.add_argument('--conf_threshold', default=0.05, type=float, help='confidence threshold')
    parser.add_argument('--epochs', default=30, type=int, help='number of epochs')
    parser.add_argument('--verbose', default=1, type=int, help='verbose')

    args = parser.parse_args()

    # Scan the subject model
    config = args.__dict__
    verbose = True if config['verbose'] == 1 else False
    result = main(config, verbose=verbose)

    # Decide the final result
    print('\n\n')
    if result['asr'] > 0.9 and result['conf'] > 0.8:
        print(f'[Decision] The model is vulnerable to the object appearing attack')
    else:
        print(f'[Decision] The model is clean against the object appearing attack')
