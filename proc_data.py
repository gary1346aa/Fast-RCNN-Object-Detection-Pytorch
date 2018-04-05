import numpy as np
import pickle
from PIL import Image
import tqdm
from tqdm import trange
from utils import *

### Train

train_pkl_path = 'data/train_data.pkl'
data = pickle.load(open(train_pkl_path, 'rb'))

train_imgs = []
train_img_info = []
train_roi = []
train_cls = []
train_tbbox = []

N_train = len(data)
for i in trange(N_train):
    img_path = data['image_name'][i]
    gt_boxs = data['boxes'][i]
    gt_classes = data['gt_classes'][i]
    # gt_overlaps = data['gt_overlaps'][i]
    # flipped = data['flipped'][i]
    nobj = data['num_objs'][i]
    bboxs = data['selective_search_boxes'][i]
    nroi = len(bboxs)

    img = Image.open('data/JPEGImages/' + img_path)
    img_size = img.size
    # print(img_size)
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32)
    img = np.transpose(img, [2, 0, 1])

    rbboxs = rel_bbox(img_size, bboxs)
    ious = calc_ious(bboxs, gt_boxs)
    max_ious = ious.max(axis=1)
    max_idx = ious.argmax(axis=1)
    tbbox = bbox_transform(bboxs, gt_boxs[max_idx])

    pos_idx = []
    neg_idx = []

    for j in range(nroi):
        if max_ious[j] < 0.1:
            continue

        gid = len(train_roi)
        train_roi.append(rbboxs[j])
        train_tbbox.append(tbbox[j])

        if max_ious[j] >= 0.5:
            pos_idx.append(gid)
            train_cls.append(gt_classes[max_idx[j]])
        else:
            neg_idx.append(gid)
            train_cls.append(0)

    pos_idx = np.array(pos_idx)
    neg_idx = np.array(neg_idx)
    train_imgs.append(img)
    train_img_info.append({
        'img_size': img_size,
        'pos_idx': pos_idx,
        'neg_idx': neg_idx,
    })
    # print(len(pos_idx), len(neg_idx))

train_imgs = np.array(train_imgs)
train_img_info = np.array(train_img_info)
train_roi = np.array(train_roi)
train_cls = np.array(train_cls)
train_tbbox = np.array(train_tbbox).astype(np.float32)

print(train_imgs.shape)
print(train_roi.shape, train_cls.shape, train_tbbox.shape)

np.savez(open('data/train.npz', 'wb'), 
         train_imgs=train_imgs, train_img_info=train_img_info,
         train_roi=train_roi, train_cls=train_cls, train_tbbox=train_tbbox)

### Test

test_pkl_path = 'data/test_data.pkl'
data = pickle.load(open(test_pkl_path, 'rb'))

test_imgs = []
test_img_info = []
test_roi = []
test_orig_roi = []

N_test = len(data)
for i in trange(N_test):
    img_path = data['image_name'][i]
    bboxs = data['selective_search_boxes'][i]
    nroi = len(bboxs)

    img = Image.open('data/JPEGImages/' + img_path)
    img_size = img.size
    # print(img_size)
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32)
    img = np.transpose(img, [2, 0, 1])

    rbboxs = rel_bbox(img_size, bboxs)
    idxs = []

    for j in range(nroi):
        gid = len(test_roi)
        test_roi.append(rbboxs[j])
        test_orig_roi.append(bboxs[j])
        idxs.append(gid)

    idxs = np.array(idxs)
    test_imgs.append(img)
    test_img_info.append({
        'img_size': img_size,
        'idxs': idxs
    })
    # print(len(idxs))

test_imgs = np.array(test_imgs)
test_img_info = np.array(test_img_info)
test_roi = np.array(test_roi)
test_orig_roi = np.array(test_orig_roi)

print(test_imgs.shape)
print(test_roi.shape)
print(test_orig_roi.shape)

np.savez(open('data/test.npz', 'wb'), 
         test_imgs=test_imgs, test_img_info=test_img_info, test_roi=test_roi, test_orig_roi=test_orig_roi)
