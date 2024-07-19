import numpy as np
import torch
import torch.nn as nn
from utils import calc_iou

class FocalLoss(nn.Module):
    @staticmethod
    def forward(classifications, regressions, anchors, annotations, std, femb_outs=None):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        femb_losses = []

        device = anchors.device
        anchor = anchors[0, :, :]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        sum_targets = torch.zeros(3)

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            femb_out = femb_outs[j, :, :] if femb_outs is not None else None
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            classification = torch.clamp(classification, 1e-3, 1.0 - 1e-3)
            if bbox_annotation.shape[0] == 0:
                alpha_factor = torch.ones(classification.shape, device=device) * alpha
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                bce = -(torch.log(1.0 - classification))
                cls_loss = focal_weight * bce
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0, dtype=torch.float, device=device))
                femb_losses.append(torch.tensor(0, dtype=torch.float, device=device))
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
            tag_max, tag_argmax = torch.max(IoU, dim=0) # num_annotations
            tag_argmax = tag_argmax[torch.ge(tag_max, 0.1)]

            # compute the loss for classification
            positive_indices = torch.ge(IoU_max, 0.5)
            sum_targets[0] += torch.tensor(torch.unique(IoU_argmax[positive_indices]).shape[0])

            positive_tag_indices = torch.zeros_like(positive_indices)
            positive_tag_indices[tag_argmax] = True
            positive_indices = positive_indices | positive_tag_indices
            sum_targets[1] += torch.tensor(torch.unique(IoU_argmax[positive_indices]).shape[0])

            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            sum_targets[2] += torch.tensor(bbox_annotation.shape[0])
            # assert sum_targets[1] >= sum_targets[0], "warn: sum_targets[1] < sum_targets[0]"
            # if torch.tensor(torch.unique(IoU_argmax[positive_indices]).shape[0]) < torch.tensor(bbox_annotation.shape[0]):
            #     print(torch.unique(IoU_argmax[positive_indices]), "||", bbox_annotation.shape[0])

            targets = torch.zeros(classification.shape, device=device)
            targets[positive_indices, 0] = 1
            alpha_factor = torch.ones(targets.shape, device=device) * alpha
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            cls_loss = focal_weight * bce
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            if num_positive_anchors == 0:
                regression_losses.append(torch.tensor(0, dtype=torch.float, device=device))
                femb_losses.append(torch.tensor(0, dtype=torch.float, device=device))
            else:
                if femb_out is not None:
                    femb_label = assigned_annotations[positive_indices, 4].to(torch.long)
                    femb_loss = nn.functional.cross_entropy(femb_out[positive_indices], femb_label)
                else:
                    femb_loss = torch.tensor(0.0)
                femb_losses.append(femb_loss)

                assigned_annotations = assigned_annotations[positive_indices, :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh)).T
                targets = targets / std

                regression_diff = torch.abs(targets - regression[positive_indices, :])
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
            torch.stack(regression_losses).mean(dim=0, keepdim=True), \
            torch.stack(femb_losses).mean(dim=0, keepdim=True)



