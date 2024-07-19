import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import cv2
import time
import numpy as np
from JDE_YoloRes import jde_yolo_resnet18
from dataloader import HeadTrackingDataset, RandSampler, DataLoader, FixCollater, RandomRatioResizer, RandomCrop, ColorJitter
from utils import calc_iou, sorted_intersect_index

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True

def train(train_loader, net, optimizer, scaler, gradient_accumulation_steps):
    epoch_loss = []
    running_loss = 0
    running_cls_loss = 0
    running_reg_loss = 0
    running_emb_loss = 0
    count = 0
    accum_knt = 0
    iter_num = 0

    net.train()
    iter_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    for i, data in enumerate(train_loader, 0):
        imgs = data['img'].to('cuda', non_blocking=True).float()
        targets = data['annot'].to('cuda', non_blocking=True)
        with autocast():
            classification_loss, regression_loss, femb_loss = net([imgs, targets])
            classification_loss /= gradient_accumulation_steps
            regression_loss /= gradient_accumulation_steps
            femb_loss /= gradient_accumulation_steps
            loss = classification_loss + regression_loss + femb_loss
        if bool(loss == 0): continue
        scaler.scale(loss).backward()

        accum_knt += 1
        running_loss += loss.item()
        running_cls_loss += classification_loss.item()
        running_reg_loss += regression_loss.item()
        running_emb_loss += femb_loss.item()
        epoch_loss.append(loss.item() * gradient_accumulation_steps)
        if accum_knt % gradient_accumulation_steps == 0:
            count += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            iter_num += 1
            if iter_num % 50 == 0:
                print(f"    {iter_num:5d} "
                        f"loss: {running_loss / count:.3f}, "
                        f"lcls: {running_cls_loss / count:.3f}, "
                        f"lreg: {running_reg_loss / count:.3f}, "
                        f"lemb: {running_emb_loss / count:.3f}, "
                        f"consume: {time.time() - iter_time:.3f}s")
                running_loss = 0
                running_cls_loss = 0
                running_reg_loss = 0
                running_emb_loss = 0
                count = 0
                iter_time = time.time()

    return np.mean(epoch_loss)

def validate(valid_loader, net, cm_loss):
    running_loss = 0
    running_cls_loss = 0
    running_reg_loss = 0
    count = 0

    net.eval()
    iter_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            imgs = data['img'].to('cuda', non_blocking=True).float()
            targets = data['annot'].to('cuda', non_blocking=True)
            with autocast():
                classification_loss, regression_loss, _ = net([imgs, targets], eval=True)
                loss = classification_loss + regression_loss

            running_loss += loss.item()
            running_cls_loss += classification_loss.item()
            running_reg_loss += regression_loss.item()
            count += 1

    print(f"    valid loss: {running_loss / count:.3f}, "
                    f"lcls: {running_cls_loss / count:.3f}, "
                    f"lreg: {running_reg_loss / count:.3f}, "
                    f"cm_loss: {cm_loss:.3f}, "
                    f"consume: {time.time() - iter_time:.3f}s")
    return running_loss / count

def calc_cm_loss(net, data_path, eval_set, step=10):
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
    cm_score_list = []
    for i in range(0, len(imgs), step):
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

        # assign a box to a target
        IoU = calc_iou(targets[:, 2:], boxes)
        IoU_max, IoU_argmax = torch.max(IoU, dim=1)
        positive_indices = torch.ge(IoU_max, 0.5)
        assigned_targets = targets[positive_indices, :]
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
            cm_score_list.append(mx_score)
        pre_info = [assigned_targets, assigned_embeds]
    return 1 - np.mean(cm_score_list)

def main(arch, num_epoch, batch_size, gradient_accumulation_steps, data_path, load_model, load_path, save_model, model_path, trained_epoch, eval_sets):
    train_dataset = HeadTrackingDataset(data_path, data_sets, transform=transforms.Compose([RandomRatioResizer(1.25), RandomCrop(512), ColorJitter()]))
    train_sampler = RandSampler(train_dataset, batch_size=batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, num_workers=4, pin_memory=True, collate_fn=FixCollater, batch_sampler=train_sampler)
    id_knt = train_dataset.id_knt
    print(f"total train sample: {len(train_dataset)}, total train id: {id_knt}")

    val_dataset = HeadTrackingDataset(data_path, eval_sets, transform=transforms.Compose([RandomRatioResizer(1.0), RandomCrop(512)]))
    val_sampler = RandSampler(val_dataset, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_dataset, num_workers=4, pin_memory=True, collate_fn=FixCollater, batch_sampler=val_sampler)
    print(f"total valid sample: {len(val_dataset)}, total valid id: {val_dataset.id_knt}")

    if load_model: load_path = model_path
    net = arch(pretrained=True, model_path=load_path, id_knt=id_knt)
    net = net.cuda()
    net = torch.compile(net)

    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[80, 85], gamma=0.1)
    scaler = GradScaler()

    for epoch in range(trained_epoch):
        optimizer.step()
        scheduler.step()
    for epoch in range(trained_epoch, num_epoch):
        begin_time = time.time()
        print(f"epoch: {epoch}, lr: {optimizer.param_groups[0]['lr']:.7f}")

        train_loss = train(train_loader, net, optimizer, scaler, gradient_accumulation_steps)
        if epoch % 1 == 0:
            valid_cm_loss = calc_cm_loss(net, data_path, eval_sets[0])
            valid_loss = validate(val_loader, net, valid_cm_loss)

        scheduler.step()
        train_sampler.shuffle(epoch)

        print(f'epoch: {epoch}, consume: {time.time() - begin_time:.3f}s, train_loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}, cm_loss {valid_cm_loss:.3f}')
        if save_model: torch.save({'state_dict': net.state_dict()}, model_path)

if __name__ == "__main__":
    load_model = False
    save_model = True
    trained_epoch = 0
    batch_size = 16
    gradient_accumulation_steps = 8
    num_epoch = 86
    arch = jde_yolo_resnet18
    data_path = '/home/zhaibowen/HeadTracking21'
    data_sets = ['HT21-01', 'HT21-02', 'HT21-03']
    eval_sets = ['HT21-04']
    load_path = '/home/zhaibowen/vision/detection/checkpoint/yolo_resnet18_mAP32.0.pth.tar'
    model_path = '/home/zhaibowen/vision/tracking/checkpoint/jde_yolo_resnet18.pth.tar'

    main(arch, num_epoch, batch_size, gradient_accumulation_steps, data_path, load_model, load_path, save_model, model_path, trained_epoch, eval_sets)


