import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import cv2
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from JDE_YoloRes import jde_yolo_resnet18
from utils import draw_caption, calc_iou, sorted_intersect_index
from dataloader import HeadTrackingDataset, RandSampler, DataLoader, FixCollater, RandomRatioResizer, RandomCrop, ColorJitter

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True

def plot_image(targets, image, scores, boxes):
    for t in targets:
        id = int(t[1])
        x1 = int(t[2])
        y1 = int(t[3])
        x2 = int(t[4])
        y2 = int(t[5])

        draw_caption(image, (x1, y1, x2, y2), str(id), bias=15)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    for j in range(scores.shape[0]):
        bbox = boxes[j, :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        draw_caption(image, (x1, y1, x2, y2), f"{scores[j]:.2f}", bias=-5)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

    print(f"targets: {targets.shape[0]}, preds: {scores.shape[0]}")
    cv2.imshow('frame', image)
    cv2.waitKey(0)

def main(arch, data_path, model_path, eval_set):
    net = arch(pretrained=True, model_path=model_path, id_knt=0)
    net = net.cuda()
    # net = torch.compile(net)
    net.eval()
    
    img_dir = os.path.join(data_path, eval_set, 'img1')
    imgs = os.listdir(img_dir)
    imgs.sort()

    gt_path = os.path.join(data_path, eval_set, 'gt/gt.txt')
    with open(gt_path, 'r') as f:
        gt = f.readlines()
        gt = list(map(lambda x: list(map(lambda x: float(x), x.strip().split(',')[:6])), gt))
        gt = np.array(gt)

    pre_info = []
    for i in range(0, len(imgs), 10):
        image_file = os.path.join(data_path, eval_set, 'img1', imgs[i])
        image = cv2.imread(image_file)
        
        image2 = np.zeros([1088, 1920, 3], dtype=np.float32)
        image2[:1080] = image
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = torch.from_numpy(image2/255.)
        image2 = torch.stack([image2]).permute(0, 3, 1, 2).contiguous()

        with torch.no_grad():
            with autocast():
                scores, embeds, boxes = net(image2.to('cuda').float(), inference=True)
                scores, embeds, boxes = scores.cpu(), embeds.cpu(), boxes.cpu()

        targets = gt[gt[:,0]==(i+1)]
        targets = torch.from_numpy(targets)
        targets[:, 4] += targets[:, 2]
        targets[:, 5] += targets[:, 3]

        plot = False
        analyse = True
        plot_mx = True
        if plot: plot_image(targets, image, scores, boxes)
        if analyse:
            # assign a box to a target
            IoU = calc_iou(targets[:, 2:], boxes)
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)
            positive_indices = torch.ge(IoU_max, 0.5)
            assigned_targets = targets[positive_indices, :]
            assigned_boxes = boxes[IoU_argmax, :][positive_indices, :]
            assigned_scores = scores[IoU_argmax][positive_indices]
            # plot_image(assigned_targets, image, assigned_scores, assigned_boxes)
            assigned_targets = assigned_targets[:, 1].numpy().astype(np.int32)
            assigned_embeds = embeds[IoU_argmax, :][positive_indices, :].to(torch.float32).numpy()
            if i > 0:
                pre_targets, pre_embeds = pre_info
                mx, my = sorted_intersect_index(pre_targets, assigned_targets)
                pre_embeds = pre_embeds[mx]
                pre_embeds = pre_embeds / np.linalg.norm(pre_embeds, axis=1, keepdims=True)
                assigned_embeds2 = assigned_embeds[my]
                assigned_embeds2 = assigned_embeds2 / np.linalg.norm(assigned_embeds2, axis=1, keepdims=True)
                confuse_matrix = np.matmul(pre_embeds, assigned_embeds2.T)
                row_max = confuse_matrix.max(axis=0)
                col_max = confuse_matrix.max(axis=1)
                diagonal_elem = np.diag(confuse_matrix)
                mx_score = (np.maximum(row_max, col_max) == diagonal_elem).sum()/diagonal_elem.shape[0]
                print(f"confuse matrix score: {mx_score:.3f}")
                if plot_mx:
                    plt.imshow(confuse_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                    plt.title('Confusion Matrix')
                    plt.colorbar()
                    # for i, j in itertools.product(range(confuse_matrix.shape[0]), range(confuse_matrix.shape[1])):
                    #     plt.text(j, i, format(int(confuse_matrix[i, j]*100), 'd'), horizontalalignment="center")
                    plt.xlabel('Predicted label')
                    plt.ylabel('True label')
                    plt.tight_layout()
                    plt.show()
            pre_info = [assigned_targets, assigned_embeds]

if __name__ == "__main__":
    arch = jde_yolo_resnet18
    data_path = '/home/work/disk/HeadTracking21/train'
    eval_set = 'HT21-04'
    model_path = '/home/work/disk/vision/tracking/checkpoint/jde_yolo_resnet18_loss0.8.pth.tar'

    main(arch, data_path, model_path, eval_set)
