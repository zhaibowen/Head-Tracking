import os
import cv2
import torch
import numpy as np
from torch.cuda.amp import autocast
from JDE_YoloRes import jde_yolo_resnet18
from track_manage import TrackManager
from utils import draw_caption
torch.backends.cudnn.benchmark = True

def infer(image, net):
    image2 = np.zeros([1088, 1920, 3], dtype=np.float32)
    image2[:1080] = image
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2 = torch.from_numpy(image2/255.)
    image2 = torch.stack([image2]).permute(0, 3, 1, 2).contiguous()

    with torch.no_grad():
        with autocast():
            scores, embeds, boxes = net(image2.to('cuda').float(), inference=True)
            scores, embeds, boxes = scores.cpu().numpy(), embeds.cpu().numpy(), boxes.cpu().numpy()
    return scores, embeds, boxes

def main(arch, data_set, model_path, fps):
    net = arch(pretrained=True, model_path=model_path, id_knt=0)
    net = net.cuda()
    net.eval()
    # net = torch.compile(net)
    
    img_dir = os.path.join(data_set, 'img1')
    imgs = os.listdir(img_dir)
    imgs.sort()

    if 'test' not in data_set:
        gt_path = os.path.join(data_set, 'gt/gt.txt')
        with open(gt_path, 'r') as f:
            gt = f.readlines()
            gt = list(map(lambda x: list(map(lambda x: float(x), x.strip().split(',')[:6])), gt))
            gt = np.array(gt)

    tracks = TrackManager(fps)

    for i in range(0, len(imgs), 5):
        image_file = os.path.join(data_set, 'img1', imgs[i])
        image = cv2.imread(image_file)
        # scores [n], embeds [n, 128], boxes [n, 4]
        scores, embeds, boxes = infer(image, net)
        tracks.update(scores, embeds, boxes)

        if 'test' not in data_set:
            targets = gt[gt[:,0]==(i+1)]
            targets = torch.from_numpy(targets)
            targets[:, 4] += targets[:, 2]
            targets[:, 5] += targets[:, 3]

            for t in targets:
                id = int(t[1])
                x1 = int(t[2])
                y1 = int(t[3])
                x2 = int(t[4])
                y2 = int(t[5])

                # draw_caption(image, (x1, y1, x2, y2), str(id), bias=15)
                # cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        for track in tracks.stable_tracks:
            id = track.id
            box = track.boxes[-1]
            box_score = track.box_scores[-1]
            score = track.scores[-1]
            mse_score = track.mse_scores[-1]
            cos_score = track.cos_scores[-1]

            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])                     
            y2 = int(box[3])

            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
            draw_caption(image, (x1, y1, x2, y2), f"{id}:"+f"{box_score:.2f}:"[2:]+f"{cos_score:.2f}"[2:], bias=-5)
            # draw_caption(image, (x1, y1, x2, y2), f"{score:.2f}", bias=-35)
            # draw_caption(image, (x1, y1, x2, y2), f"{mse_score:.2f}", bias=-50)
            # draw_caption(image, (x1, y1, x2, y2), f"{cos_score:.2f}", bias=-65)

        if 'test' in data_set:
            print(f"preds: {len(tracks.stable_tracks)}")
        else:
            print(f"targets: {targets.shape[0]}, preds: {len(tracks.stable_tracks)}")
        cv2.imshow('frame', image)
        cv2.waitKey(0)

if __name__ == "__main__":
    fps = 5
    arch = jde_yolo_resnet18
    # data_set = '/home/work/disk/HeadTracking21/train/HT21-01'
    data_set = '/home/work/disk/HeadTracking21/test/HT21-11'
    # model_path = '/home/work/disk/vision/tracking/checkpoint/jde_yolo_resnet18.pth.tar'
    model_path = '/home/work/disk/vision/tracking/checkpoint/jde_yolo_resnet18_loss0.557.pth.tar'

    main(arch, data_set, model_path, fps)