import os
import cv2
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from utils import draw_caption, rand_scale, rand_uniform_strong

def assert_truth_target(anchor, area, min_inter=0.6):
    x1, y1, x2, y2 = anchor
    if x1 > area[2] or x2 < area[0] or y1 > area[3] or y2 < area[1]:
        return False
    inter = (min(x2, area[2]) - max(x1, area[0])) * (min(y2, area[3]) - max(y1, area[1]))
    area = (x2 - x1) * (y2 - y1)
    if inter / area < min_inter:
        return False
    return True

def crop_anchor(anchor, area):
    x1, y1, x2, y2 = anchor
    x1 = max(x1, area[0])
    y1 = max(y1, area[1])
    x2 = min(x2, area[2])
    y2 = min(y2, area[3])
    return [x1, y1, x2, y2]

class HeadTrackingDataset(Dataset):
    def __init__(self, data_path, data_sets, transform=None):
        # image origin shape [1920, 1080]
        # 一张图片分成6份, shape [640, 540]
        self.transform= transform
        self.images = []
        self.id_knt = 0

        id_step=100000
        col_step, row_step = 640, 540
        cols = [0, 640, 1280, 1920]
        rows = [0, 540, 1080]
        id_map = {}
        for i, data_set in enumerate(data_sets):
            img_dir = os.path.join(data_path, data_set, 'img1')
            imgs = os.listdir(img_dir)
            imgs = imgs * 6
            imgs.sort()
            for j, img in enumerate(imgs):
                cid = (j % 6) // 2
                rid = (j % 6) % 2
                imgs[j] = [os.path.join(data_path, data_set, 'img1', img), [cols[cid], rows[rid], cols[cid+1], rows[rid+1]], []]

            gt_path = os.path.join(data_path, data_set, 'gt/gt.txt')
            with open(gt_path, 'r') as f:
                gt = f.readlines()
                gt = list(map(lambda x: list(map(lambda x: float(x), x.strip().split(',')[:6])), gt))

            for label in gt:
                iid, uid, x1, y1, w, h = label
                x2, y2 = x1 + w, y1 + h
                cx, cy = x1 + w / 2, y1 + h / 2
                pid = 6 * int(iid - 1) + 2 * int(cx / col_step) + int(cy / row_step)
                area = imgs[pid][1]
                if not assert_truth_target([x1, y1, x2, y2], area):
                    continue
                x1, y1, x2, y2 = crop_anchor([x1, y1, x2, y2], area)
                # re-id
                mid = i * id_step + int(uid)
                if mid not in id_map:
                    id_map[mid] = self.id_knt
                    self.id_knt += 1
                imgs[pid][2].append([x1 - area[0], y1 - area[1], x2 - area[0], y2 - area[1], id_map[mid]])

            self.images.extend(imgs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_file, area, annot = self.images[idx]
        image = cv2.imread(image_file)
        image = image[area[1] : area[3], area[0] : area[2]]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annot = np.array(annot, dtype=np.float32)
        sample = {'img': image, 'annot': annot, 'scale': 1.0}
        if self.transform:
            sample = self.transform(sample)
        img, annot, scale = sample['img'], sample['annot'], sample['scale']
        return {'img': torch.from_numpy(img/255.), 'annot': torch.from_numpy(annot), 'scale': scale}

class RandomRatioResizer(object):
    def __init__(self, scale=1.25):
        self.scale = scale

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        rows, cols, _ = image.shape
        scale = rand_scale(self.scale)
        image = cv2.resize(image, (int(round((cols*scale))), int(round(rows*scale))))
        if annots.shape[0]:
            annots[:, :4] *= scale
        return {'img': image, 'annot': annots, 'scale': scale}
    
class RandomCrop(object):
    def __init__(self, size=512):
        self.size = size

    def __call__(self, sample):
        image, annots, scale = sample['img'], sample['annot'], sample['scale']
        img = np.zeros((self.size, self.size, 3), dtype=np.float32)

        rows, cols, _ = image.shape
        if rows >= self.size and cols >= self.size:
            x = random.randint(0, cols - self.size)
            y = random.randint(0, rows - self.size)
            img = image[y : y + self.size, x : x + self.size]
            if annots.shape[0]:
                annots[:, [0, 2]] -= x
                annots[:, [1, 3]] -= y
        elif rows >= self.size and cols < self.size:
            x = random.randint(0, self.size - cols)
            y = random.randint(0, rows - self.size)
            img[:, x : x + cols] = image[y : y + self.size, :]
            if annots.shape[0]:
                annots[:, [0, 2]] += x
                annots[:, [1, 3]] -= y
        elif rows < self.size and cols >= self.size:
            x = random.randint(0, cols - self.size)
            y = random.randint(0, self.size - rows)
            img[y : y + rows, :] = image[:, x : x + self.size]
            if annots.shape[0]:
                annots[:, [0, 2]] -= x
                annots[:, [1, 3]] += y
        else:
            x = random.randint(0, self.size - cols)
            y = random.randint(0, self.size - rows)
            img[y : y + rows, x : x + cols] = image
            if annots.shape[0]:
                annots[:, [0, 2]] += x
                annots[:, [1, 3]] += y

        area = [0, 0, 512, 512]
        if annots.shape[0]:
            annots = annots[list(map(lambda x: assert_truth_target(x[:4], area), annots.tolist()))]
        if annots.shape[0]:
            annots[:, :4] = np.array(list(map(lambda x: crop_anchor(x[:4], area), annots.tolist())))
        return {'img': img, 'annot': annots, 'scale': scale}

class ColorJitter(object):
    def __init__(self, hue=0.1, saturation=1.5, exposure=1.5):
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def __call__(self, sample):
        img, annots, scale = sample['img'], sample['annot'], sample['scale']
        rows, cols, _ = img.shape

        dhue = rand_uniform_strong(-self.hue, self.hue)
        dsat = rand_scale(self.saturation)
        dexp = rand_scale(self.exposure)
        flip = random.randint(0, 1)

        if dsat != 1 or dexp != 1 or dhue != 0:
            hsv_src = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2HSV)  # RGB to HSV
            hsv = list(cv2.split(hsv_src))
            hsv[1] *= dsat
            hsv[2] *= dexp
            hsv[0] += 179 * dhue
            hsv_src = cv2.merge(hsv)
            img = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)  # HSV to RGB (the same as previous)

        if flip:
            img = cv2.flip(img, 1)
            if annots.shape[0]:
                temp = cols - annots[:, 0]
                annots[:, 0] = cols - annots[:, 2]
                annots[:, 2] = temp
        
        return {'img': img, 'annot': annots, 'scale': scale}

def FixCollater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = torch.stack(imgs).permute(0, 3, 1, 2).contiguous()
    return {'img': imgs, 'annot': annot_padded, 'scale': scales}

class RandSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, shuffle=True):
        self.batch_size = batch_size
        self.order = list(range(len(data_source)))
        self.total_size = len(self.order) - len(self.order) % self.batch_size
        if not drop_last: self.total_size += batch_size

        if shuffle: random.shuffle(self.order)
        self.groups = []
        for i in range(0, self.total_size, self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def shuffle(self, epoch=0):
        random.shuffle(self.order)
        self.groups = []
        for i in range(0, self.total_size, self.batch_size):
            self.groups.append([self.order[x % len(self.order)] for x in range(i, i + self.batch_size)])

    def __iter__(self):
        for group in self.groups:
            yield group

    def __len__(self):
        return len(self.groups)

if __name__ == "__main__":
    data_path = '/home/work/disk/HeadTracking21/train'
    data_sets = ['HT21-01', 'HT21-04']
    train_dataset = HeadTrackingDataset(data_path, data_sets, transform=transforms.Compose([RandomRatioResizer(1.25), RandomCrop(512), ColorJitter()]))
    train_sampler = RandSampler(train_dataset, batch_size=8, drop_last=True)
    train_loader = DataLoader(train_dataset, num_workers=0, pin_memory=True, collate_fn=FixCollater, batch_sampler=train_sampler)

    for i, data in enumerate(train_loader, 0):
        img, annot = data['img'][0], data['annot'][0]
        img = img.permute(1, 2, 0).contiguous()
        img = img.numpy() * 255
        img[img<0] = 0
        img[img>255] = 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        annot = annot[annot[:, 0] >= 0]
        annot = annot.tolist()
        for t in annot:
            x1 = int(t[0])
            y1 = int(t[1])
            x2 = int(t[2])
            y2 = int(t[3])
            id = int(t[4])
            draw_caption(img, (x1, y1, x2, y2), str(id), bias=15)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        cv2.imshow('frame', img)
        cv2.waitKey(0)
        a = 1

    # x=list(filter(lambda x: 'HT21-04' in x[0] and 640 == x[1][0] and 540 == x[1][1], train_dataset.images))
    # for image_file, area, targets in x:
    #     # 1080, 1920, 3
    #     image = cv2.imread(image_file)
    #     image = image[area[1] : area[3], area[0] : area[2]]
    #     for t in targets:
    #         id = int(t[1])
    #         x1 = int(t[2])
    #         y1 = int(t[3])
    #         x2 = int(t[4])
    #         y2 = int(t[5])
    #         draw_caption(image, (x1, y1, x2, y2), str(id), bias=15)
    #         cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    #     cv2.imshow('frame', image)
    #     cv2.waitKey(1)