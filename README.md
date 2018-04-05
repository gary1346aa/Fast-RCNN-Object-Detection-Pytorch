{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Fast-RCNN\n",
    "------\n",
    "We didn't upload the score of this part because the RCNN related model usually needs pretraining to obtain good accuracy. However, the dataset for pretraining is too large for our computer and network speed, but we are curious how the pretrained model can perform so we decided to complete this model but with downloaded pretrain model which is illegal. In the following section, we'll use *PyTorch* instead of Tensorflow to implement the process.\n",
    "\n",
    "![Alt text](https://i.imgur.com/0gAX9ku.png \"Optional title\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Preprocessing\n",
    "------\n",
    "The dataset given is preprocessed with **selective search data**, but the post-processing code wasn't really useful, so we re-write it by ourself."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import sys\n",
    "import pickle\n",
    "import tqdm\n",
    "import torch\n",
    "import torchvision\n",
    "import torch.nn as nn\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from torch.autograd import Variable\n",
    "from PIL import Image\n",
    "from tqdm import trange\n",
    "from utils import *\n",
    "from roi import *"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "First we load the given selective search data :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train_pkl_path = 'data/train_data.pkl'\n",
    "data = pickle.load(open(train_pkl_path, 'rb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Then we collect the features stored in the given pickle to according arrays, and resize every images to **224x224**, so every picture become small and having same size, which is simple and efficient for training."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train_imgs = []\n",
    "train_img_info = []\n",
    "train_roi = []\n",
    "train_cls = []\n",
    "train_tbbox = []\n",
    "\n",
    "N_train = len(data)\n",
    "for i in trange(N_train):\n",
    "    img_path = data['image_name'][i]\n",
    "    gt_boxs = data['boxes'][i]\n",
    "    gt_classes = data['gt_classes'][i]\n",
    "    nobj = data['num_objs'][i]\n",
    "    bboxs = data['selective_search_boxes'][i]\n",
    "    nroi = len(bboxs)\n",
    "\n",
    "    img = Image.open('data/JPEGImages/' + img_path)\n",
    "    img_size = img.size\n",
    "    img = img.resize((224, 224))\n",
    "    img = np.array(img).astype(np.float32)\n",
    "    img = np.transpose(img, [2, 0, 1])\n",
    "\n",
    "    rbboxs = rel_bbox(img_size, bboxs)\n",
    "    ious = calc_ious(bboxs, gt_boxs)\n",
    "    max_ious = ious.max(axis=1)\n",
    "    max_idx = ious.argmax(axis=1)\n",
    "    tbbox = bbox_transform(bboxs, gt_boxs[max_idx])\n",
    "\n",
    "    pos_idx = []\n",
    "    neg_idx = []\n",
    "\n",
    "    for j in range(nroi):\n",
    "        if max_ious[j] < 0.1:\n",
    "            continue\n",
    "\n",
    "        gid = len(train_roi)\n",
    "        train_roi.append(rbboxs[j])\n",
    "        train_tbbox.append(tbbox[j])\n",
    "\n",
    "        if max_ious[j] >= 0.5:\n",
    "            pos_idx.append(gid)\n",
    "            train_cls.append(gt_classes[max_idx[j]])\n",
    "        else:\n",
    "            neg_idx.append(gid)\n",
    "            train_cls.append(0)\n",
    "\n",
    "    pos_idx = np.array(pos_idx)\n",
    "    neg_idx = np.array(neg_idx)\n",
    "    train_imgs.append(img)\n",
    "    train_img_info.append({\n",
    "        'img_size': img_size,\n",
    "        'pos_idx': pos_idx,\n",
    "        'neg_idx': neg_idx,\n",
    "    })\n",
    "    #print(len(pos_idx), len(neg_idx))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Converting them to `np.array()`, and then save the transformed data into a new *.npz* file."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "train_imgs = np.array(train_imgs)\n",
    "train_img_info = np.array(train_img_info)\n",
    "train_roi = np.array(train_roi)\n",
    "train_cls = np.array(train_cls)\n",
    "train_tbbox = np.array(train_tbbox).astype(np.float32)\n",
    "\n",
    "print(f'Training image dataset shape : {train_imgs.shape}')\n",
    "print(f'ROI : {train_roi.shape}, Train_cls : {train_cls.shape}, Train_tbbox : {train_tbbox.shape}')\n",
    "\n",
    "np.savez(open('data/train.npz', 'wb'), \n",
    "         train_imgs=train_imgs, train_img_info=train_img_info,\n",
    "         train_roi=train_roi, train_cls=train_cls, train_tbbox=train_tbbox)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The preprocessing of the testing data is quite the same."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "test_pkl_path = 'data/test_data.pkl'\n",
    "data = pickle.load(open(test_pkl_path, 'rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "test_imgs = []\n",
    "test_img_info = []\n",
    "test_roi = []\n",
    "test_orig_roi = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "N_test = len(data)\n",
    "for i in trange(N_test):\n",
    "    img_path = data['image_name'][i]\n",
    "    bboxs = data['selective_search_boxes'][i]\n",
    "    nroi = len(bboxs)\n",
    "\n",
    "    img = Image.open('data/JPEGImages/' + img_path)\n",
    "    img_size = img.size\n",
    "    img = img.resize((224, 224))\n",
    "    img = np.array(img).astype(np.float32)\n",
    "    img = np.transpose(img, [2, 0, 1])\n",
    "\n",
    "    rbboxs = rel_bbox(img_size, bboxs)\n",
    "    idxs = []\n",
    "\n",
    "    for j in range(nroi):\n",
    "        gid = len(test_roi)\n",
    "        test_roi.append(rbboxs[j])\n",
    "        test_orig_roi.append(bboxs[j])\n",
    "        idxs.append(gid)\n",
    "\n",
    "    idxs = np.array(idxs)\n",
    "    test_imgs.append(img)\n",
    "    test_img_info.append({\n",
    "        'img_size': img_size,\n",
    "        'idxs': idxs\n",
    "    })"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "test_imgs = np.array(test_imgs)\n",
    "test_img_info = np.array(test_img_info)\n",
    "test_roi = np.array(test_roi)\n",
    "test_orig_roi = np.array(test_orig_roi)\n",
    "\n",
    "print(f'Testing image dataset shape : {test_imgs.shape}')\n",
    "\n",
    "np.savez(open('data/test.npz', 'wb'), \n",
    "         test_imgs=test_imgs, test_img_info=test_img_info, test_roi=test_roi, test_orig_roi=test_orig_roi)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Utilities\n",
    "------\n",
    "Before training, we construct some utility functions for convenience, most of them are from the given lecture/hint codes:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def rel_bbox(size, bbox):\n",
    "    bbox = bbox.astype(np.float32)\n",
    "    bbox[:,0] /= size[0]\n",
    "    bbox[:,1] /= size[1]\n",
    "    bbox[:,2] += 1\n",
    "    bbox[:,2] /= size[0]\n",
    "    bbox[:,3] += 1\n",
    "    bbox[:,3] /= size[1]\n",
    "    return bbox"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def bbox_transform(ex_rois, gt_rois):\n",
    "    ex_widths = ex_rois[:,2] - ex_rois[:,0] + 1.0\n",
    "    ex_heights = ex_rois[:,3] - ex_rois[:,1] + 1.0\n",
    "    ex_ctr_x = ex_rois[:,0] + 0.5 * ex_widths\n",
    "    ex_ctr_y = ex_rois[:,1] + 0.5 * ex_heights\n",
    "\n",
    "    gt_widths = gt_rois[:,2] - gt_rois[:,0] + 1.0\n",
    "    gt_heights = gt_rois[:,3] - gt_rois[:,1] + 1.0\n",
    "    gt_ctr_x = gt_rois[:,0] + 0.5 * gt_widths\n",
    "    gt_ctr_y = gt_rois[:,1] + 0.5 * gt_heights\n",
    "\n",
    "    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths\n",
    "    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights\n",
    "    targets_dw = np.log(gt_widths / ex_widths)\n",
    "    targets_dh = np.log(gt_heights / ex_heights)\n",
    "\n",
    "    targets = np.array([targets_dx, targets_dy, targets_dw, targets_dh]).T\n",
    "    return targets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def calc_ious(ex_rois, gt_rois):\n",
    "    ex_area = (1. + ex_rois[:,2] - ex_rois[:,0]) * (1. + ex_rois[:,3] - ex_rois[:,1])\n",
    "    gt_area = (1. + gt_rois[:,2] - gt_rois[:,0]) * (1. + gt_rois[:,3] - gt_rois[:,1])\n",
    "    area_sum = ex_area.reshape((-1, 1)) + gt_area.reshape((1, -1))\n",
    "\n",
    "    lb = np.maximum(ex_rois[:,0].reshape((-1, 1)), gt_rois[:,0].reshape((1, -1)))\n",
    "    rb = np.minimum(ex_rois[:,2].reshape((-1, 1)), gt_rois[:,2].reshape((1, -1)))\n",
    "    tb = np.maximum(ex_rois[:,1].reshape((-1, 1)), gt_rois[:,1].reshape((1, -1)))\n",
    "    ub = np.minimum(ex_rois[:,3].reshape((-1, 1)), gt_rois[:,3].reshape((1, -1)))\n",
    "\n",
    "    width = np.maximum(1. + rb - lb, 0.)\n",
    "    height = np.maximum(1. + ub - tb, 0.)\n",
    "    area_i = width * height\n",
    "    area_u = area_sum - area_i\n",
    "    ious = area_i / area_u\n",
    "    return ious"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def reg_to_bbox(img_size, reg, box):\n",
    "    img_width, img_height = img_size\n",
    "    bbox_width = box[:,2] - box[:,0] + 1.0\n",
    "    bbox_height = box[:,3] - box[:,1] + 1.0\n",
    "    bbox_ctr_x = box[:,0] + 0.5 * bbox_width\n",
    "    bbox_ctr_y = box[:,1] + 0.5 * bbox_height\n",
    "\n",
    "    bbox_width = bbox_width[:,np.newaxis]\n",
    "    bbox_height = bbox_height[:,np.newaxis]\n",
    "    bbox_ctr_x = bbox_ctr_x[:,np.newaxis]\n",
    "    bbox_ctr_y = bbox_ctr_y[:,np.newaxis]\n",
    "\n",
    "    out_ctr_x = reg[:,:,0] * bbox_width + bbox_ctr_x\n",
    "    out_ctr_y = reg[:,:,1] * bbox_height + bbox_ctr_y\n",
    "\n",
    "    out_width = bbox_width * np.exp(reg[:,:,2])\n",
    "    out_height = bbox_height * np.exp(reg[:,:,3])\n",
    "\n",
    "    return np.array([\n",
    "        np.maximum(0, out_ctr_x - 0.5 * out_width),\n",
    "        np.maximum(0, out_ctr_y - 0.5 * out_height),\n",
    "        np.minimum(img_width, out_ctr_x + 0.5 * out_width),\n",
    "        np.minimum(img_height, out_ctr_y + 0.5 * out_height)\n",
    "    ]).transpose([1, 2, 0])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following non_maximum_suppression() filters the boxes that are overlapped."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def non_maximum_suppression(sc, bboxs, iou_threshold=0.7, score_threshold=0.6):\n",
    "    nroi = sc.shape[0]\n",
    "    idx = np.argsort(sc)[::-1]\n",
    "    rb = 0\n",
    "    while rb < nroi and sc[idx[rb]] >= score_threshold:\n",
    "        rb += 1\n",
    "    if rb == 0:\n",
    "        return []\n",
    "    idx = idx[:rb]\n",
    "    sc = sc[idx]\n",
    "    bboxs = bboxs[idx,:]\n",
    "    ious = calc_ious(bboxs, bboxs)\n",
    "\n",
    "    res = []\n",
    "    for i in range(rb):\n",
    "        if i == 0 or ious[i, :i].max() < iou_threshold:\n",
    "            res.append(bboxs[i])\n",
    "\n",
    "    return res"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def plot (name, title, legendx, legendy, x, y, n_epoch, frame_size = 256,labelx = 'Epoch', labely = 'Loss'):\n",
    "    i = 0\n",
    "    x = np.array(x).flatten('F')\n",
    "    y = np.array(y).flatten('F')\n",
    "    framex = []\n",
    "    framey = []\n",
    "    \n",
    "    while i*frame_size < len(x):\n",
    "        framex.append(np.mean(x[i*frame_size:min(len(x),(i+1)*frame_size)]))\n",
    "        framey.append(np.mean(y[i*frame_size:min(len(y),(i+1)*frame_size)]))\n",
    "        i += 1\n",
    "    \n",
    "    a = np.arange(0,len(x),len(x)/len(framex))\n",
    "    b = a/len(y)*n_epoch\n",
    "    a = a/len(x)*n_epoch\n",
    "    \n",
    "    plt.figure()\n",
    "    plt.plot(a,framex)\n",
    "    plt.plot(b,framey)\n",
    "    plt.xlabel(labelx)\n",
    "    plt.ylabel(labely)\n",
    "    plt.title(title)\n",
    "    plt.legend([legendx,legendy])\n",
    "    plt.savefig(name,dpi=600)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Model : Fast-RCNN\n",
    "------\n",
    "Our model is constructed based on the Fast-RCNN model, which is introduced by Ross Girshick in 2015. Generally speaking, RCNN is implemented through four following steps :\n",
    "\n",
    "1. Obtain about 1000~2000 selective search boxes in each data (which is done in the given dataset)\n",
    "2. Apply Convolutional newral network to each selective search boxes, and obtain features\n",
    "3. Apply Classifier to the features obtained in the CNN model to classify if it belongs to any category.\n",
    "4. Apply Regression to refine the search boxes into more accurate position"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "But the problem is that RCNN performs at a super slow speed, how slow it is? Considering the given TA's code, the CNN model is fed by the complete (resized) image, and 1 epoch takes about 75 seconds with GTX1080Ti, how about feeding the model with selective search boxes which is a 1000 to 2000 times larger dataset? Then 1 epoch will take about 20 hours, which is too slow for our hardware and deadline."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So we come to Fast-RCNN, and the model flowchart is shown above :\n",
    "![Alt text](http://img.blog.csdn.net/20160411214438672 \"Optional title\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "-----\n",
    "As we can see, the model is a type of CNN, with RELU and Max_pooling. For the fifth pooling, the layer is changed with **ROI pooling**, which divide every selective search boxes into *MxN* sections evenly, and apply max-pooling for each sections. Finally, ROI pooling transfer the vary sized selective search data into a uniformed data, and send them to the next layer.\n",
    "The code is shown below:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class SlowROIPool(nn.Module):\n",
    "    def __init__(self, output_size):\n",
    "        super().__init__()\n",
    "        self.maxpool = nn.AdaptiveMaxPool2d(output_size)\n",
    "        self.size = output_size\n",
    "\n",
    "    def forward(self, images, rois, roi_idx):\n",
    "        n = rois.shape[0]\n",
    "        h = images.size(2)\n",
    "        w = images.size(3)\n",
    "        x1 = rois[:,0]\n",
    "        y1 = rois[:,1]\n",
    "        x2 = rois[:,2]\n",
    "        y2 = rois[:,3]\n",
    "\n",
    "        x1 = np.floor(x1 * w).astype(int)\n",
    "        x2 = np.ceil(x2 * w).astype(int)\n",
    "        y1 = np.floor(y1 * h).astype(int)\n",
    "        y2 = np.ceil(y2 * h).astype(int)\n",
    "        \n",
    "        res = []\n",
    "        for i in range(n):\n",
    "            img = images[roi_idx[i]].unsqueeze(0)\n",
    "            img = img[:, :, y1[i]:y2[i], x1[i]:x2[i]]\n",
    "            img = self.maxpool(img)\n",
    "            res.append(img)\n",
    "        res = torch.cat(res, dim=0)\n",
    "        return res"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In order to implement the model with better performance, we use the VGG16 model, which is the same as the structure used in Fast-RCNN paper, and we switch the last pooling layer into our modified ROI-pooling layer.\n",
    "\n",
    "![Alt text](https://junzhangcom.files.wordpress.com/2017/08/vgg16.png \"Optional title\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "class RCNN(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "\n",
    "        rawnet = torchvision.models.vgg16_bn(pretrained=True)\n",
    "        self.seq = nn.Sequential(*list(rawnet.features.children())[:-1])\n",
    "        self.roipool = SlowROIPool(output_size=(7, 7))\n",
    "        self.feature = nn.Sequential(*list(rawnet.classifier.children())[:-1])\n",
    "\n",
    "        _x = Variable(torch.Tensor(1, 3, 224, 224))\n",
    "        _r = np.array([[0., 0., 1., 1.]])\n",
    "        _ri = np.array([0])\n",
    "        _x = self.feature(self.roipool(self.seq(_x), _r, _ri).view(1, -1))\n",
    "        feature_dim = _x.size(1)\n",
    "        self.cls_score = nn.Linear(feature_dim, N_CLASS+1)\n",
    "        self.bbox = nn.Linear(feature_dim, 4*(N_CLASS+1))\n",
    "        \n",
    "        self.cel = nn.CrossEntropyLoss()\n",
    "        self.sl1 = nn.SmoothL1Loss()\n",
    "\n",
    "    def forward(self, inp, rois, ridx):\n",
    "        res = inp\n",
    "        res = self.seq(res)\n",
    "        res = self.roipool(res, rois, ridx)\n",
    "        res = res.detach()\n",
    "        res = res.view(res.size(0), -1)\n",
    "        feat = self.feature(res)\n",
    "\n",
    "        cls_score = self.cls_score(feat)\n",
    "        bbox = self.bbox(feat).view(-1, N_CLASS+1, 4)\n",
    "        return cls_score, bbox\n",
    "\n",
    "    def calc_loss(self, probs, bbox, labels, gt_bbox):\n",
    "        loss_sc = self.cel(probs, labels)\n",
    "        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 4)\n",
    "        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4)\n",
    "        loss_loc = self.sl1(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox * mask)\n",
    "        lmb = 1.0\n",
    "        loss = loss_sc + lmb * loss_loc\n",
    "        return loss, loss_sc, loss_loc"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To accelerate the processing progress, we compute in *cuda()* instead of computing in CPU, and we can see the structure in the following block:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RCNN (\n",
      "  (seq): Sequential (\n",
      "    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (2): ReLU (inplace)\n",
      "    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (5): ReLU (inplace)\n",
      "    (6): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))\n",
      "    (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (9): ReLU (inplace)\n",
      "    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (12): ReLU (inplace)\n",
      "    (13): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))\n",
      "    (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (16): ReLU (inplace)\n",
      "    (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (19): ReLU (inplace)\n",
      "    (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (22): ReLU (inplace)\n",
      "    (23): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))\n",
      "    (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (26): ReLU (inplace)\n",
      "    (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (29): ReLU (inplace)\n",
      "    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (32): ReLU (inplace)\n",
      "    (33): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))\n",
      "    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (36): ReLU (inplace)\n",
      "    (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (39): ReLU (inplace)\n",
      "    (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n",
      "    (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)\n",
      "    (42): ReLU (inplace)\n",
      "  )\n",
      "  (roipool): SlowROIPool (\n",
      "    (maxpool): AdaptiveMaxPool2d (output_size=(7, 7))\n",
      "  )\n",
      "  (feature): Sequential (\n",
      "    (0): Linear (25088 -> 4096)\n",
      "    (1): ReLU (inplace)\n",
      "    (2): Dropout (p = 0.5)\n",
      "    (3): Linear (4096 -> 4096)\n",
      "    (4): ReLU (inplace)\n",
      "    (5): Dropout (p = 0.5)\n",
      "  )\n",
      "  (cls_score): Linear (4096 -> 21)\n",
      "  (bbox): Linear (4096 -> 84)\n",
      "  (cel): CrossEntropyLoss (\n",
      "  )\n",
      "  (sl1): SmoothL1Loss (\n",
      "  )\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "N_CLASS = 20\n",
    "\n",
    "Transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "\n",
    "rcnn = RCNN().cuda()\n",
    "print(rcnn)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "-----\n",
    "We load the preprocessed data, disorganize and split them into train and test data with ratio 4:1. The optimizer used in the training process is Adam stochastic optimization."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "npz = np.load('data/train.npz')\n",
    "train_imgs = npz['train_imgs']\n",
    "train_img_info = npz['train_img_info']\n",
    "train_roi = npz['train_roi']\n",
    "train_cls = npz['train_cls']\n",
    "train_tbbox = npz['train_tbbox']\n",
    "\n",
    "train_imgs = torch.from_numpy(train_imgs)\n",
    "train_imgs = Transform(train_imgs)\n",
    "\n",
    "Ntotal = train_imgs.size(0)\n",
    "Ntrain = int(Ntotal * 0.8)\n",
    "pm = np.random.permutation(Ntotal)\n",
    "train_set = pm[:Ntrain]\n",
    "val_set = pm[Ntrain:]\n",
    "\n",
    "optimizer = torch.optim.Adam(rcnn.parameters(), lr=1e-4)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following functions defines the training batches generation, epoch training/validation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=False):\n",
    "    sc, r_bbox = rcnn(img, rois, ridx)\n",
    "    loss, loss_sc, loss_loc = rcnn.calc_loss(sc, r_bbox, gt_cls, gt_tbbox)\n",
    "    fl = loss.data.cpu().numpy()[0]\n",
    "    fl_sc = loss_sc.data.cpu().numpy()[0]\n",
    "    fl_loc = loss_loc.data.cpu().numpy()[0]\n",
    "\n",
    "    if not is_val:\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "    return fl, fl_sc, fl_loc\n",
    "\n",
    "def train_epoch(run_set, is_val=False):\n",
    "    I = 2\n",
    "    B = 64\n",
    "    POS = int(B * 0.25)\n",
    "    NEG = B - POS\n",
    "    Nimg = len(run_set)\n",
    "    perm = np.random.permutation(Nimg)\n",
    "    perm = run_set[perm]\n",
    "    losses = []\n",
    "    losses_sc = []\n",
    "    losses_loc = []\n",
    "    \n",
    "    for i in trange(0, Nimg, I):\n",
    "        lb = i\n",
    "        rb = min(i+I, Nimg)\n",
    "        torch_seg = torch.from_numpy(perm[lb:rb])\n",
    "        img = Variable(train_imgs[torch_seg], volatile=is_val).cuda()\n",
    "        ridx = []\n",
    "        glo_ids = []\n",
    "\n",
    "        for j in range(lb, rb):\n",
    "            info = train_img_info[perm[j]]\n",
    "            pos_idx = info['pos_idx']\n",
    "            neg_idx = info['neg_idx']\n",
    "            ids = []\n",
    "\n",
    "            if len(pos_idx) > 0:\n",
    "                ids.append(np.random.choice(pos_idx, size=POS))\n",
    "            if len(neg_idx) > 0:\n",
    "                ids.append(np.random.choice(neg_idx, size=NEG))\n",
    "            if len(ids) == 0:\n",
    "                continue\n",
    "            ids = np.concatenate(ids, axis=0)\n",
    "            glo_ids.append(ids)\n",
    "            ridx += [j-lb] * ids.shape[0]\n",
    "\n",
    "        if len(ridx) == 0:\n",
    "            continue\n",
    "        glo_ids = np.concatenate(glo_ids, axis=0)\n",
    "        ridx = np.array(ridx)\n",
    "\n",
    "        rois = train_roi[glo_ids]\n",
    "        gt_cls = Variable(torch.from_numpy(train_cls[glo_ids]), volatile=is_val).cuda()\n",
    "        gt_tbbox = Variable(torch.from_numpy(train_tbbox[glo_ids]), volatile=is_val).cuda()\n",
    "\n",
    "        loss, loss_sc, loss_loc = train_batch(img, rois, ridx, gt_cls, gt_tbbox, is_val=is_val)\n",
    "        losses.append(loss)\n",
    "        losses_sc.append(loss_sc)\n",
    "        losses_loc.append(loss_loc)\n",
    "\n",
    "    avg_loss = np.mean(losses)\n",
    "    avg_loss_sc = np.mean(losses_sc)\n",
    "    avg_loss_loc = np.mean(losses_loc)\n",
    "    print(f'Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')\n",
    "    \n",
    "    return losses, losses_sc, losses_loc"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Then we define the process with 1 training epoch with 1 validation epoch. Besides training, we store the loss during the training process and plot them with our plotting function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def start_training(n_epoch=2):\n",
    "    tl = []\n",
    "    ts = []\n",
    "    to = []\n",
    "    vl = []\n",
    "    vs = [] \n",
    "    vo = []\n",
    "    for i in range(n_epoch):\n",
    "        print(f'===========================================')\n",
    "        print(f'[Training Epoch {i+1}]')\n",
    "        train_loss, train_sc, train_loc = train_epoch(train_set, False)\n",
    "        print(f'[Validation Epoch {i+1}]')\n",
    "        val_loss, val_sc, val_loc = train_epoch(val_set, True)\n",
    "        \n",
    "        tl.append(train_loss)\n",
    "        ts.append(train_sc)\n",
    "        to.append(train_loc)\n",
    "        vl.append(val_loss)\n",
    "        vs.append(val_sc)\n",
    "        vo.append(val_loc)\n",
    "        \n",
    "    plot('loss','Train/Val : Loss', 'Train', 'Validation', tl, vl, n_epoch)\n",
    "    plot('loss_sc','Train/Val : Loss_sc', 'Train', 'Validation', ts, vs, n_epoch)\n",
    "    plot('loss_loc','Train/Val : Loss_loc', 'Train', 'Validation', to, vo, n_epoch)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Testing process is nearly the same as training, while the boxes is calculated mainly from `non_maximum_suppression()`, which is introduced in the utilities."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "npz = np.load('data/test.npz')\n",
    "test_imgs = npz['test_imgs']\n",
    "test_img_info = npz['test_img_info']\n",
    "test_roi = npz['test_roi']\n",
    "test_orig_roi = npz['test_orig_roi']\n",
    "\n",
    "test_imgs = torch.from_numpy(test_imgs)\n",
    "test_imgs = Transform(test_imgs).cuda()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def test_image(img, img_size, rois, orig_rois):\n",
    "    nroi = rois.shape[0]\n",
    "    ridx = np.zeros(nroi).astype(int)\n",
    "    sc, tbbox = rcnn(img, rois, ridx)\n",
    "    sc = nn.functional.softmax(sc)\n",
    "    sc = sc.data.cpu().numpy()\n",
    "    tbbox = tbbox.data.cpu().numpy()\n",
    "    bboxs = reg_to_bbox(img_size, tbbox, orig_rois)\n",
    "\n",
    "    res_bbox = []\n",
    "    res_cls = []\n",
    "\n",
    "    for c in range(1, N_CLASS+1):\n",
    "        c_sc = sc[:,c]\n",
    "        c_bboxs = bboxs[:,c,:]\n",
    "\n",
    "        boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0.3, score_threshold=0.6)\n",
    "        res_bbox.extend(boxes)\n",
    "        res_cls.extend([c] * len(boxes))\n",
    "\n",
    "    if len(res_cls) == 0:\n",
    "        for c in range(1, N_CLASS+1):\n",
    "            c_sc = sc[:,c]\n",
    "            c_bboxs = bboxs[:,c,:]\n",
    "\n",
    "            boxes = non_maximum_suppression(c_sc, c_bboxs, iou_threshold=0.3, score_threshold=0.3)\n",
    "            res_bbox.extend(boxes)\n",
    "            res_cls.extend([c] * len(boxes))\n",
    "        res_bbox = res_bbox[:1]\n",
    "        res_cls = res_cls[:1]\n",
    "\n",
    "    print(res_cls)\n",
    "\n",
    "    return np.array(res_bbox), np.array(res_cls)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Finally, we define the testing process and send the final generated data into the given `evaluate()` function to obtain the final *output.csv* file."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "def test_epoch():\n",
    "    Nimg = test_imgs.size(0)\n",
    "    Nc = Nimg\n",
    "\n",
    "    perm = np.random.permutation(Nimg)[:Nc]\n",
    "\n",
    "    bbox_preds = []\n",
    "    bbox_cls = []\n",
    "\n",
    "    for i in range(Nimg):\n",
    "        bbox_preds.append(np.ndarray((0, 4)))\n",
    "        bbox_cls.append(np.ndarray((0, 1)))\n",
    "\n",
    "    for i in range(Nc):\n",
    "        pi = perm[i]\n",
    "        img = Variable(test_imgs[pi:pi+1], volatile=True)\n",
    "        ridx = []\n",
    "        glo_ids = []\n",
    "\n",
    "        info = test_img_info[pi]\n",
    "        img_size = info['img_size']\n",
    "        idxs = info['idxs']\n",
    "\n",
    "        idxs = np.array(idxs)\n",
    "        rois = test_roi[idxs]\n",
    "        orig_rois = test_orig_roi[idxs]\n",
    "\n",
    "        res_bbox, res_cls = test_image(img, img_size, rois, orig_rois)\n",
    "        bbox_preds[pi] = res_bbox\n",
    "        bbox_cls[pi] = res_cls\n",
    "\n",
    "    evaluate.evaluate(bbox_preds, bbox_cls)\n",
    "\n",
    "    print('Test complete')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "1. For initial training, comment the following loading line, uncomment the training/saving lines.\n",
    "2. For continual training, uncomment all of the following line.\n",
    "3. For pure testing, uncomment the loading line, comment the training/saving lines.\n",
    "\n",
    "Observing the loss graph, we can see that due to pretraining, the loss is already low in the initial training epoch, and is about to well-fit in the third epoch."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\r",
      "  0%|          | 0/4159 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "===========================================\n",
      "[Training Epoch 1]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 4159/4159 [04:26<00:00, 15.60it/s]\n",
      "  0%|          | 3/1040 [00:00<00:42, 24.62it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Avg loss = 0.7839; loss_sc = 0.7696, loss_loc = 0.0143\n",
      "[Validation Epoch 1]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1040/1040 [00:27<00:00, 37.91it/s]\n",
      "  0%|          | 2/4159 [00:00<03:30, 19.79it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Avg loss = 0.6817; loss_sc = 0.6728, loss_loc = 0.0089\n",
      "===========================================\n",
      "[Training Epoch 2]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 4159/4159 [04:31<00:00, 15.33it/s]\n",
      "  0%|          | 3/1040 [00:00<00:39, 26.44it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Avg loss = 0.6183; loss_sc = 0.6101, loss_loc = 0.0082\n",
      "[Validation Epoch 2]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1040/1040 [00:27<00:00, 38.40it/s]\n",
      "  0%|          | 3/4159 [00:00<03:32, 19.53it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Avg loss = 0.6560; loss_sc = 0.6485, loss_loc = 0.0074\n",
      "===========================================\n",
      "[Training Epoch 3]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 4159/4159 [04:26<00:00, 15.59it/s]\n",
      "  0%|          | 3/1040 [00:00<00:38, 26.85it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Avg loss = 0.5486; loss_sc = 0.5411, loss_loc = 0.0075\n",
      "[Validation Epoch 3]\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 1040/1040 [00:26<00:00, 38.52it/s]\n",
      "/home/gary/anaconda3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2909: RuntimeWarning: Mean of empty slice.\n",
      "  out=out, **kwargs)\n",
      "/home/gary/anaconda3/lib/python3.6/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in true_divide\n",
      "  ret = ret.dtype.type(ret / rcount)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Avg loss = 0.6561; loss_sc = 0.6492, loss_loc = 0.0069\n"
     ]
    },
    {
     "data": {
      "image/png": "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\nRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fedf41e8898>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n80rfu2QgmTJlChEREZW+XnXSQFKOmDBN3KiUsoYOHco333zDkSP282DDhg1s3bqV5ORkBgwYQEpK\nCl27duWrr7467twNGzbQpUsXAHJychg+fDjdunXj8ssvJycnp/i4W2+9tTj9/BNPPAHAK6+8wtat\nW+nXrx/9+vUDoFWrVuzcuROAMWPG0KVLF7p06cJLL71UfL9OnTpx880307lzZ84555xj7lOddGGr\nckQ5+ba0w12pOmbqw7B9WfVes1lXGFz6yhTR0dH07NmTadOmcdFFFzF+/Hguv/xygoODmTRpEo0a\nNWLnzp2ceuqpXHjhhaWuhf7GG28QEhLC0qVLWbp0KSkpKcX7nnnmGaKioigoKGDAgAEsXbqUu+66\nizFjxjBz5kxiYmKOudZvv/3G2LFjWbBgAcYYevXqRZ8+fYiMjGTt2rV88sknvPXWWwwbNoyJEydy\n1VVXVc/vyoXWSMoRXZxKXgOJUurY5q2iZi1jDI8++ijdunVj4MCBbNmyhR07dpR6jTlz5hR/oHfr\n1o1u3boV75swYQIpKSn06NGDFStWuE0/72ru3LlccsklhIaGEhYWxqWXXspPP/0EQGJiIsnJyYB3\n09RrjaQcUZpvS6m6qYyagzddfPHF3HfffSxatIicnBxSUlJ47733yMrK4rfffsPf359WrVq5TRvv\nyl1tZf369Tz//PMsXLiQyMhIrrvuunKvU9aI0sDAwOLXvr6+Xmva0hpJOUICfAn089FAopQC7Cis\nvn37csMNNxR3su/bt48mTZrg7+/PzJkz2bhxY5nXOOussxg3bhwAy5cvZ+nSpYBNPx8aGkrjxo3Z\nsWMHU6dOLT4nPDycAwcOuL3Wl19+SXZ2NocOHWLSpEmceeaZ1fW4HtEaSTlExKZJ0VFbSinHiBEj\nuPTSS4ubuK688kqGDBlCamoqycnJdOzYsczzb731Vq6//nq6detGcnIyPXvadfu6d+9Ojx496Ny5\n83Hp50eOHMngwYNp3rw5M2fOLN6ekpLCddddV3yNm266iR49etToaovSECbapaammvLGaJflgld/\nIjYskLHX9yz/YKWU16xatYpOnTrVdjFOSO5+tyLymzEmtbxztWnLA5omRSmlSqeBxAOaAVgppUqn\ngcQDmgFYqbqjITTH17Sq/k41kHggKjSA7NwCDudpvi2lalNQUBC7du3SYFKNjDHs2rWLoKCgSl/D\n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\nAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fed9c129438>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fed9c466da0>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "#rcnn.load_state_dict(torch.load('model/hao123.mdl'))\n",
    "start_training(n_epoch=3)\n",
    "torch.save(rcnn.state_dict(), 'model/hao123.mdl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "test_epoch()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
