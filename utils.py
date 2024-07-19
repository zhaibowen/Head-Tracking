import cv2
import random
import torch
import numpy as np

def draw_caption(image, box, caption, bias=10):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] + bias), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] + bias), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)

def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min

def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU

def sorted_intersect_index(x, y): 
    '''
        x, y是一维无序数组，且每个值都是唯一的
        求出x, y的排序交集
        返回交集中元素对应x, y的索引，如
            x = np.array([4, 1, 10, 5, 8, 13, 11]) 
            y = np.array([20, 5, 4, 9, 11, 7, 25]) 
            它们的排序交集是 [4, 5, 11]
            mx = np.array([0, 3, 6]) 
            my = np.array([2, 1, 4]) 
    '''
    aux = np.concatenate((x, y)) 
    sidx = aux.argsort() 
    inidx = aux[sidx[1:]] == aux[sidx[:-1]] 
 
    xym = np.vstack((sidx[inidx.nonzero()], sidx[1:][inidx.nonzero()])).T.flatten() 
    xm = xym[xym < len(x)] 
    ym = xym[xym >= len(x)] - len(x) 
    return xm, ym 