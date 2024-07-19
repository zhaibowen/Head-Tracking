import os
import time
import numpy as np
from torchvision import transforms
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer, random_center_initializer
from pyclustering.utils.metric import distance_metric, type_metric
from dataloader import HeadTrackingDataset, RandSampler, DataLoader, FixCollater, RandomRatioResizer, RandomCrop, ColorJitter

def calc_iou(a, b):
    iw = np.minimum(a[0], b[0])
    ih = np.minimum(a[1], b[1])
    ua = a[0] * a[1] + b[0] * b[1] - iw * ih
    intersection = iw * ih
    IoU = intersection / ua
    return 1 - IoU

def cluster():
    cluster_num = 5

    data_path = '/home/work/disk/HeadTracking21/train'
    data_sets = ['HT21-01', 'HT21-02', 'HT21-03', 'HT21-04']
    train_dataset = HeadTrackingDataset(data_path, data_sets, transform=transforms.Compose([RandomRatioResizer(1.25), RandomCrop(512), ColorJitter()]))
    train_sampler = RandSampler(train_dataset, batch_size=128, drop_last=True)
    train_loader = DataLoader(train_dataset, num_workers=4, pin_memory=True, collate_fn=FixCollater, batch_sampler=train_sampler)

    annots = []
    for i, data in enumerate(train_loader, 0):
        print(i)
        annot = data['annot'].reshape(-1, 5)
        annot = annot[annot[:, 0] >= 0]
        annot = annot[:, :4].numpy()
        annots.append(annot)
        # if i > 6: break

    ctime = time.time()
    annots = np.concatenate(annots)
    points = np.zeros([annots.shape[0], 2])
    points[:, 0] = annots[:, 2] - annots[:, 0]
    points[:, 1] = annots[:, 3] - annots[:, 1]
    print(points.shape[0])

    metric = distance_metric(type_metric.USER_DEFINED, func=calc_iou, numpy_usage=False)
    initial_centers = kmeans_plusplus_initializer(points, cluster_num).initialize()
    kmeans_instance = kmeans(points, initial_centers, metric=metric)
    kmeans_instance.process()

    final_anchors = []
    final_centers = kmeans_instance.get_centers()
    final_clusters = kmeans_instance.get_clusters()

    for j in range(cluster_num):
        final_centers[j].append(len(final_clusters[j]))

    final_centers = np.array(final_centers)
    final_centers = final_centers[(final_centers[:, 0] * final_centers[:, 1]).argsort()]
    final_centers = final_centers.astype(np.int32).tolist()
    print(final_centers)

    for j in range(cluster_num):
        final_anchors.append(final_centers[j][:2])

    print(time.time() - ctime)
    print(final_anchors)



if __name__ == "__main__":
    cluster()

# 895970 anchor
# [[18, 18, 154092], [23, 23, 254325], [29, 29, 244332], [35, 35, 173542], [44, 45, 69679]]
# [[18, 18], [23, 23], [29, 29], [35, 35], [44, 45]]