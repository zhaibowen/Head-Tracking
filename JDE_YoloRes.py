import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.ops import nms
from anchors import YoloAnchors
from focal_loss import FocalLoss

class BBoxTransform(nn.Module):
    @staticmethod
    def forward(img, boxes, deltas, std):
        widths  = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * std[0, 0]
        dy = deltas[:, :, 1] * std[0, 1]
        dw = deltas[:, :, 2] * std[0, 2]
        dh = deltas[:, :, 3] * std[0, 3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        batch_size, num_channels, height, width = img.shape
        boxes[:, :, 0] = torch.clamp(pred_boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(pred_boxes[:, :, 1], min=0)
        boxes[:, :, 2] = torch.clamp(pred_boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(pred_boxes[:, :, 3], max=height)
        return boxes

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class Conv_Batch_Active(nn.Module):
    def __init__(self, cin, out, kernel, stride=1, padding=0, bn=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, out, kernel, stride, padding),
            nn.BatchNorm2d(out) if bn else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class BasicBlock(nn.Module):
    def __init__(self, cin, out, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, out, 3, stride, 1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True),
            nn.Conv2d(out, out, 3, 1, 1),
            nn.BatchNorm2d(out)
        )

        self.shortcut = nn.Identity() if cin == out and stride == 1 else \
                        nn.Sequential(nn.Conv2d(cin, out, 3, stride, 1),
                                      nn.BatchNorm2d(out)
                        )

    def forward(self, x):
        return functional.relu(self.block(x) + self.shortcut(x), inplace=True)

class SPP(nn.Module):
    def __init__(self, cin):
        super().__init__()
        out = cin
        self.conv1 = Conv_Batch_Active(cin, out, 1, bn=True)
        self.maxpool1 = nn.MaxPool2d(3, 1, 3//2)
        self.maxpool2 = nn.MaxPool2d(5, 1, 5//2)
        self.maxpool3 = nn.MaxPool2d(7, 1, 7//2)
        self.conv3 = Conv_Batch_Active(out*4, out, 1, bn=True)

    def forward(self, x):
        x = self.conv1(x)
        m1 = self.maxpool1(x)
        m2 = self.maxpool2(x)
        m3 = self.maxpool3(x)
        spp = torch.cat([m3, m2, m1, x], dim=1)
        x = self.conv3(spp)
        return x

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, C6_size, feature_size=256):
        super().__init__()

        self.P6_1 = Conv_Batch_Active(C6_size, feature_size, 1, bn=True)
        self.P6_2 = Conv_Batch_Active(feature_size, feature_size, 3, 1, 1, bn=True)

        self.P6_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_1 = Conv_Batch_Active(C5_size, feature_size, 1, bn=True)
        self.P5_2 = Conv_Batch_Active(feature_size*2, feature_size, 3, 1, 1, bn=True)

        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_1 = Conv_Batch_Active(C4_size, feature_size, 1, bn=True)
        self.P4_2 = Conv_Batch_Active(feature_size*2, feature_size, 3, 1, 1, bn=True)

        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_1 = Conv_Batch_Active(C3_size, feature_size, 1, bn=True)
        self.P3_2 = Conv_Batch_Active(feature_size*2, feature_size, 3, 1, 1, bn=True)

    def forward(self, inputs):
        C3, C4, C5, C6 = inputs

        P6_x = self.P6_1(C6)
        P6_x = self.P6_2(P6_x)

        P6_upsampled_x = self.P6_upsampled(P6_x)
        P5_x = self.P5_1(C5)
        s, t = P5_x.shape[-2], P5_x.shape[-1]
        P5_x = torch.cat([P6_upsampled_x[:, :, :s, :t], P5_x], dim=1)
        P5_x = self.P5_2(P5_x)

        P5_upsampled_x = self.P5_upsampled(P5_x)
        P4_x = self.P4_1(C4)
        P4_x = torch.cat([P5_upsampled_x, P4_x], dim=1)
        P4_x = self.P4_2(P4_x)

        P4_upsampled_x = self.P4_upsampled(P4_x)
        P3_x = self.P3_1(C3)
        P3_x = torch.cat([P3_x, P4_upsampled_x], dim=1)
        P3_x = self.P3_2(P3_x)

        return P3_x

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=5, feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors
        self.conv1 = Conv_Batch_Active(num_features_in, feature_size, 3, 1, 1)
        self.conv2 = Conv_Batch_Active(feature_size, feature_size, 3, 1, 1)
        self.conv3 = Conv_Batch_Active(feature_size, feature_size, 3, 1, 1)
        self.conv4 = Conv_Batch_Active(feature_size, feature_size, 3, 1, 1)
        self.output_new = nn.Conv2d(feature_size, num_anchors * 4, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.output_new(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 4)

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=5, femb_dim=128, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.femb_dim = femb_dim
        self.num_anchors = num_anchors
        self.conv1 = Conv_Batch_Active(num_features_in, feature_size, 3, 1, 1)
        self.conv2 = Conv_Batch_Active(feature_size, feature_size, 3, 1, 1)
        self.conv3 = Conv_Batch_Active(feature_size, feature_size, 3, 1, 1)
        self.conv4 = Conv_Batch_Active(feature_size, feature_size, 3, 1, 1)
        self.output_new = nn.Conv2d(feature_size, num_anchors, 3, 1, 1)
        self.output_femb = nn.Conv2d(feature_size, femb_dim, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out_c = torch.sigmoid(self.output_new(out))
        out_c = out_c.permute(0, 2, 3, 1)
        out_c = out_c.contiguous().view(x.shape[0], -1, 1)

        out_f = self.output_femb(out)
        out_f = out_f.permute(0, 2, 3, 1)
        out_f = out_f.contiguous().view(x.shape[0], -1, self.femb_dim)

        return out_c, out_f

class ResNet(nn.Module):
    def __init__(self, block, layers, channels, id_knt, device):
        super(ResNet, self).__init__()
        num_anchors = 5
        femb_dim = 128
        self.device = device
        self.num_anchors = num_anchors

        # 3, 512
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # 64, 256
        self.conv2 = self._make_layer(block, layers[0], channels[0], channels[1], maxpool=True)
        # 64, 128
        self.conv3 = self._make_layer(block, layers[1], channels[1], channels[2])
        # 128, 64
        self.conv4 = self._make_layer(block, layers[2], channels[2], channels[3])
        # 256, 32
        self.conv5 = self._make_layer(block, layers[3], channels[3], channels[4])
        # 512, 16
        self.conv6 = self._make_layer(block, layers[4], channels[4], channels[5])
        # 512, 8

        self.spp = SPP(channels[5])
        self.fpn = PyramidFeatures(channels[2], channels[3], channels[4], channels[5], 256)
        self.regressionModel = RegressionModel(256, num_anchors=num_anchors)
        self.classificationModel = ClassificationModel(256, num_anchors=num_anchors, femb_dim=femb_dim)
        self.norm = RMSNorm(femb_dim)
        if id_knt > 0:
            self.femb_layer = nn.Linear(femb_dim, id_knt, bias=True)

        # 3, 608, 608
        self.anchors = YoloAnchors(num_anchors)
        self.regbox_std = torch.tensor([[0.1, 0.1, 0.2, 0.2]], device=device)
        self.regressBoxes = BBoxTransform()
        self.focalLoss = FocalLoss()

    @staticmethod
    def _make_layer(block, num_layer, cin, out, maxpool=False):
        if maxpool:
            layers = [nn.MaxPool2d(3, 2, 1),
                      block(cin, out)]
        else:
            layers = [block(cin, out, 2)]
        for i in range(num_layer-1):
            layers.append(block(out, out))
        return nn.Sequential(*layers)

    def forward(self, inputs, inference=False, eval=False):
        if not inference:
            x, annotations = inputs
        else:
            x = inputs
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        spp = self.spp(x6)
        p3 = self.fpn([x3, x4, x5, spp])
        regression = self.regressionModel(p3)
        classification, fembedding = self.classificationModel(p3)
        fembedding = self.norm(fembedding)
        anchors = self.anchors(x)

        if not inference:
            if eval:
                femb_out = None # do not calc fembedding loss
            else:
                femb_out = self.femb_layer(fembedding)
                femb_out = femb_out[:,:,None,:].tile(1, 1, self.num_anchors, 1).view(x.shape[0], -1, femb_out.shape[-1])
            return self.focalLoss(classification, regression, anchors, annotations, self.regbox_std, femb_out)

        transformed_anchors = self.regressBoxes(x, anchors, regression, self.regbox_std)

        finalScores = torch.Tensor([]).to(self.device)
        finalEmbeddings = torch.Tensor([]).long().to(self.device)
        finalAnchorBoxesCoordinates = torch.Tensor([]).to(self.device)

        scores = torch.squeeze(classification[:, :, 0])
        scores_over_thresh = (scores > 0.3)
        if scores_over_thresh.sum() == 0:
            return [finalScores, finalEmbeddings, finalAnchorBoxesCoordinates]

        scores = scores[scores_over_thresh]
        anchorBoxes = torch.squeeze(transformed_anchors)
        anchorBoxes = anchorBoxes[scores_over_thresh]
        anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
        
        fembedding = fembedding[:,:,None,:].tile(1, 1, self.num_anchors, 1).view(-1, fembedding.shape[-1])
        fembedding = fembedding[scores_over_thresh]

        finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
        finalEmbeddings = torch.cat((finalEmbeddings, fembedding[anchors_nms_idx]))
        finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

        return [finalScores, finalEmbeddings, finalAnchorBoxesCoordinates]

def jde_yolo_resnet18(pretrained=False, model_path=None, id_knt=0, device='cuda'):
    model = ResNet(BasicBlock, [2, 2, 2, 2, 2], [64, 64, 128, 256, 512, 512], id_knt, device)
    if pretrained:
        state_dict = torch.load(model_path, map_location=torch.device(device))['state_dict']
        replacer = '_orig_mod.'
        if "module" == list(state_dict.keys())[0][:6]:
            replacer = 'module.'
        model.load_state_dict({k.replace(replacer, ''): v for k, v in state_dict.items()}, strict=False)
    return model

if __name__ == '__main__':
    x = jde_yolo_resnet18(id_knt=1000)
    y = torch.zeros([10,3,512,512])
    targets = torch.tensor([[1,1,30,30,1], [80, 80, 110, 110, 2]], dtype=torch.float32)
    targets = torch.tile(targets[None, ...], (10,1,1)).contiguous().cuda()
    y = y.cuda()
    x = x.cuda()
    z = x((y, targets))
    a = 1