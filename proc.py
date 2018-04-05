import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from PIL import Image
from roi import *
from utils import *
import time
from tqdm import trange
import sys

sys.path.insert(0, './evaluate')
import evaluate

N_CLASS = 20

Transform = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        rawnet = torchvision.models.vgg16_bn(pretrained=True)
        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1])
        # self.roipool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        self.roipool = SlowROIPool(output_size=(7, 7))
        self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1])

        _x = Variable(torch.Tensor(1, 3, 224, 224))
        _r = np.array([[0., 0., 1., 1.]])
        _ri = np.array([0])
        _x = self.feature(self.roipool(self.seq(_x), _r, _ri).view(1, -1))
        feature_dim = _x.size(1)
        self.cls_score = nn.Linear(feature_dim, N_CLASS+1)
        self.bbox = nn.Linear(feature_dim, 4*(N_CLASS+1))
        
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, inp, rois, ridx):
        res = inp
        res = self.seq(res)
        res = self.roipool(res, rois, ridx)
        res = res.detach()
        res = res.view(res.size(0), -1)
        feat = self.feature(res)

        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat).view(-1, N_CLASS+1, 4)
        return cls_score, bbox

    def calc_loss(self, probs, bbox, labels, gt_bbox):
        loss_sc = self.cel(probs, labels)
        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 4)
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4)
        loss_loc = self.sl1(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox * mask)
        lmb = 1.0
        loss = loss_sc + lmb * loss_loc
        return loss, loss_sc, loss_loc


rcnn = RCNN().cuda()
print(rcnn)

npz = np.load('data/train.npz')
train_imgs = npz['train_imgs']
train_img_info = npz['train_img_info']
train_roi = npz['train_roi']
train_cls = npz['train_cls']
train_tbbox = npz['train_tbbox']

train_imgs = torch.from_numpy(train_imgs)
train_imgs = Transform(train_imgs)

Ntotal = train_imgs.size(0)
Ntrain = int(Ntotal * 0.8)
pm = np.random.permutation(Ntotal)
train_set = pm[:Ntrain]
val_set = pm[Ntrain:]

optimizer = torch.optim.Adam(rcnn.parameters(), lr=1e-4)

def train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=False):
    sc, r_bbox = rcnn(img, rois, ridx)
    loss, loss_sc, loss_loc = rcnn.calc_loss(sc, r_bbox, gt_cls, gt_tbbox)
    fl = loss.data.cpu().numpy()[0]
    fl_sc = loss_sc.data.cpu().numpy()[0]
    fl_loc = loss_loc.data.cpu().numpy()[0]

    if not is_val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return fl, fl_sc, fl_loc

def train_epoch(run_set, is_val=False):
    I = 2
    B = 64
    POS = int(B * 0.25)
    NEG = B - POS
    Nimg = len(run_set)
    perm = np.random.permutation(Nimg)
    perm = run_set[perm]

    # if is_val:
        # rcnn.eval()
    # else:
        # rcnn.train()

    losses = []
    losses_sc = []
    losses_loc = []
    for i in trange(0, Nimg, I):
        lb = i
        rb = min(i+I, Nimg)
        torch_seg = torch.from_numpy(perm[lb:rb])
        img = Variable(train_imgs[torch_seg], volatile=is_val).cuda()
        ridx = []
        glo_ids = []

        for j in range(lb, rb):
            info = train_img_info[perm[j]]
            pos_idx = info['pos_idx']
            neg_idx = info['neg_idx']
            ids = []

            if len(pos_idx) > 0:
                ids.append(np.random.choice(pos_idx, size=POS))
            if len(neg_idx) > 0:
                ids.append(np.random.choice(neg_idx, size=NEG))
            if len(ids) == 0:
                continue
            ids = np.concatenate(ids, axis=0)
            glo_ids.append(ids)
            ridx += [j-lb] * ids.shape[0]

        if len(ridx) == 0:
            continue
        glo_ids = np.concatenate(glo_ids, axis=0)
        ridx = np.array(ridx)

        rois = train_roi[glo_ids]
        gt_cls = Variable(torch.from_numpy(train_cls[glo_ids]), volatile=is_val).cuda()
        gt_tbbox = Variable(torch.from_numpy(train_tbbox[glo_ids]), volatile=is_val).cuda()

        loss, loss_sc, loss_loc = train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=is_val)
        losses.append(loss)
        losses_sc.append(loss_sc)
        losses_loc.append(loss_loc)

    avg_loss = np.mean(losses)
    avg_loss_sc = np.mean(losses_sc)
    avg_loss_loc = np.mean(losses_loc)
    print(f'Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')

def start_training(n_epoch=5):
    for i in range(n_epoch):
        print(f'===========================================')
        print(f'[Training Epoch {i+1}]')
        train_epoch(train_set, False)
        print(f'[Validation Epoch {i+1}]')
        train_epoch(val_set, True)

npz = np.load('data/test.npz')
test_imgs = npz['test_imgs']
test_img_info = npz['test_img_info']
test_roi = npz['test_roi']
test_orig_roi = npz['test_orig_roi']

test_imgs = torch.from_numpy(test_imgs)
test_imgs = Transform(test_imgs).cuda()

def test_image(img, img_size, rois, orig_rois):
    nroi = rois.shape[0]
    ridx = np.zeros(nroi).astype(int)
    sc, tbbox = rcnn(img, rois, ridx)
    sc = nn.functional.softmax(sc)
    sc = sc.data.cpu().numpy()
    tbbox = tbbox.data.cpu().numpy()
    bboxs = reg_to_bbox(img_size, tbbox, orig_rois)

    res_bbox = []
    res_cls = []

    for c in range(1, N_CLASS+1):
        c_sc = sc[:,c]
        c_bboxs = bboxs[:,c,:]

        boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0.3, score_threshold=0.6)
        res_bbox.extend(boxes)
        res_cls.extend([c] * len(boxes))

    if len(res_cls) == 0:
        for c in range(1, N_CLASS+1):
            c_sc = sc[:,c]
            c_bboxs = bboxs[:,c,:]

            boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0.3, score_threshold=0.3)
            res_bbox.extend(boxes)
            res_cls.extend([c] * len(boxes))
        res_bbox = res_bbox[:1]
        res_cls = res_cls[:1]

    print(res_cls)

    return np.array(res_bbox), np.array(res_cls)

def test_epoch():
    Nimg = test_imgs.size(0)
    # rcnn.eval()
    Nc = Nimg // 10

    perm = np.random.permutation(Nimg)[:Nc]

    bbox_preds = []
    bbox_cls = []

    for i in range(Nimg):
        bbox_preds.append(np.ndarray((0, 4)))
        bbox_cls.append(np.ndarray((0, 1)))

    for i in trange(Nc):
        pi = perm[i]
        img = Variable(test_imgs[pi:pi+1], volatile=True)
        ridx = []
        glo_ids = []

        info = test_img_info[pi]
        img_size = info['img_size']
        idxs = info['idxs']

        idxs = np.array(idxs)
        rois = test_roi[idxs]
        orig_rois = test_orig_roi[idxs]

        res_bbox, res_cls = test_image(img, img_size, rois, orig_rois)
        bbox_preds[pi] = res_bbox
        bbox_cls[pi] = res_cls

    evaluate.evaluate(bbox_preds, bbox_cls)

    print('Test complete')

rcnn.load_state_dict(torch.load('model/hao123.mdl'))
# start_training(n_epoch=2)
# torch.save(rcnn.state_dict(), 'model/hao123.mdl')

test_epoch()

