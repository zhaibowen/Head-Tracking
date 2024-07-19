import numpy as np
import torch
import torch.nn as nn

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

class YoloAnchors(nn.Module):
    def __init__(self, num_anchors):
        super(YoloAnchors, self).__init__()
        self.strides = [8]
        self.num_anchors = num_anchors
        self.anchors = [[18, 18], [23, 23], [29, 29], [35, 35], [44, 45]]
        self.anchors = np.array(self.anchors).reshape((1, num_anchors, 2))

    def forward(self, image):
        # 3, 512, 512
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [np.ceil(image_shape / x).astype(np.int32) for x in self.strides]
        # (8, 8)

        all_anchors = np.zeros((0, 4)).astype(np.float32)
        for idx, stride in enumerate(self.strides):
            anchors = generate_anchors_yolo(self.anchors[idx], self.num_anchors)
            shifted_anchors = shift(image_shapes[idx], stride, anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)
        return torch.from_numpy(all_anchors.astype(np.float32)).cuda()

def generate_anchors_yolo(anchor_shapes, num_anchors):
    anchors = np.zeros((num_anchors, 4))
    anchors[:, 2:] = anchor_shapes
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors