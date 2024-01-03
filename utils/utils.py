import os
import math
import glob
import shutil
import random
from itertools import groupby
from dataclasses import dataclass

import cv2
import numpy as np


# More: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/utils/asr_confidence_utils.py

@dataclass
class ConfidenceMeasure:
    method: str = 'tsallis'
    assert method in ['tsallis', 'renyi', 'gibbs'], f'Confidence measure {method} is not supported.'

    def __call__(self, x, v, t):
        neg_entropy_gibbs = lambda x: (x.exp() * x).sum(-1)
        neg_entropy_alpha = lambda x, t: (x * t).exp().sum(-1)
        neg_entropy_alpha_gibbs = lambda x, t: ((x * t).exp() * x).sum(-1)

        entropy_gibbs_lin_baseline = lambda x, v: 1 + neg_entropy_gibbs(x) / math.log(v)
        entropy_gibbs_exp_baseline = lambda x, v: (neg_entropy_gibbs(x).exp() * v - 1) / (v - 1)

        def entropy_tsallis_exp(x, v, t):
            exp_neg_max_ent = math.exp((1 - math.pow(v, 1 - t)) / (1 - t))
            return (((1 - neg_entropy_alpha(x, t)) / (1 - t)).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

        def entropy_gibbs_exp(x, v, t):
            exp_neg_max_ent = math.pow(v, -t * math.pow(v, 1 - t))
            return ((neg_entropy_alpha_gibbs(x, t) * t).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

        if self.method == 'tsallis':
            confidence_measure = (
                lambda x, v, t: entropy_gibbs_exp_baseline(x, v)
                if t == 1.0
                else entropy_tsallis_exp(x, v, t)
            )
            return confidence_measure(x, v, t)
        elif self.method == 'renyi':
            confidence_measure = (
                lambda x, v, t: entropy_gibbs_exp_baseline(x, v)
                if t == 1.0
                else (neg_entropy_alpha(x, t).pow(1 / (t - 1)) * v - 1) / (v - 1)
            )
            return confidence_measure(x, v, t)
        elif self.method == 'gibbs':
            confidence_measure = (
                lambda x, v, t: entropy_gibbs_exp_baseline(x, v)
                if t == 1.0
                else entropy_gibbs_exp(x, v, t)
            )
            return confidence_measure(x, v, t)
        else:
            raise ValueError(f'Confidence measure {self.method} is not supported.')


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def cv2_imwrite(file_path, img):
    cv2.imencode('.jpg', img)[1].tofile(file_path)


def show(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w*r), height)
    else:
        r = width / float(w)
        dim = (width, int(h*r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# center fit and support rgb img
def center_fit(img, w, h, inter=cv2.INTER_NEAREST, top_left=True):
    # get img shape
    img_h, img_w = img.shape[:2]
    # get ratio
    ratio = min(w / img_w, h / img_h)

    if len(img.shape) == 3:
        inter = cv2.INTER_AREA
    # resize img
    img = cv2.resize(img, (int(img_w * ratio), int(img_h * ratio)), interpolation=inter)
    # get new img shape
    img_h, img_w = img.shape[:2]
    # get start point
    start_w = (w - img_w) // 2
    start_h = (h - img_h) // 2

    if top_left:
        start_w = 0
        start_h = 0

    if len(img.shape) == 2:
        # create new img
        new_img = np.zeros((h, w), dtype=np.uint8)
        new_img[start_h:start_h+img_h, start_w:start_w+img_w] = img
    else:
        new_img = np.zeros((h, w, 3), dtype=np.uint8)
        new_img[start_h:start_h+img_h, start_w:start_w+img_w, :] = img

    return new_img


# load dict from txt, with number of lines
def load_dict(dict_path='data/label.names'):
    with open(dict_path, 'r', encoding='utf-8') as f:
        dict = f.read().splitlines()
    dict = {i: dict[i] for i in range(len(dict))}
    return dict


def decode_label(mat, chars) -> str:
    # mat is the output of model
    # get char indices along best path
    best_path_indices = np.argmax(mat[0], axis=-1)
    # collapse best path (using itertools.groupby), map to chars, join char list to string
    best_chars_collapsed = [chars[k] for k, _ in groupby(best_path_indices) if k != len(chars)]
    res = ''.join(best_chars_collapsed)

    # remove space and _
    res = res.replace(' ', '').replace('_', '')

    return res


# move file to dir
def move2dir(file_path, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # get file name
    file_name = os.path.basename(file_path)
    # get file new path
    new_path = os.path.join(dir_path, file_name)
    # move file, if file exists, overwrite it
    if not os.path.exists(new_path):
        os.rename(file_path, new_path)


def copy2dir(file_path, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # get file name
    file_name = os.path.basename(file_path)
    # get file new path
    new_path = os.path.join(dir_path, file_name)
    # copy file, if file exists, overwrite it
    if not os.path.exists(new_path):
        shutil.copy(file_path, new_path)


def path2label(path):
    # split label with _
    labels = os.path.splitext(os.path.basename(path))[0].split('_')
    t_labels = ""

    while len(t_labels) < 7:
        label = labels.pop(0)
        # remove space and _
        label = label.replace(' ', '').replace('_', '')
        t_labels += label

    len_ = len(t_labels)

    # if A-Z in label, len_ += 1, if a-z in label, len_ += 0
    for i in t_labels:
        if i.isupper():
            len_ += 1
            break
        elif i.islower():
            break

    return len_, t_labels


# find matched img path via given txt files
def find_img_paths(txt_paths, img_dir, img_ext='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.{}'.format(img_ext)))
    # find which both in txt_paths and img_paths
    img_paths = [img_path for img_path in img_paths if os.path.splitext(img_path)[0] + '.txt' in txt_paths]
    return img_paths


import unittest
import torch

class TestGetConfidence(unittest.TestCase):
    def test_get_confidence(self):
        # load from pred.npy
        x = torch.from_numpy(np.load('pred.npy'))[0]
        # print argmax idx and value
        idx = torch.argmax(x, dim=-1).tolist()
        idx_max = torch.max(x, dim=-1).values.tolist()
        # zip idx and value
        conf_mean = []
        for i, j in zip(idx, idx_max):
            # if idx is 85, continue
            if i >= 85: continue
            # keep 2 decimal places
            print(i, '\t {:.4f}'.format(j))
            conf_mean.append(j)

        print('mean:', np.mean(conf_mean), "\n")

        for idx, i in enumerate(x):
            # print(i)
            t = idx + 1
            gbs_exp = ConfidenceMeasure('gibbs')
            f1 = gbs_exp(i, 86, t).item()
            print(idx, '\t {:.4f}'.format(f1))

        # # remove blank lines if argmax is 85
        # new_x = []
        # idxs = torch.argmax(x, dim=-1)
        # for idx, i in enumerate(idxs):
        #     if i >= 85:
        #         continue
        #     new_x.append(x[idx])
        # x = torch.stack(new_x)

        # np.savetxt('pred.txt', x, fmt='%.4f')
        # v = 84

        # # get confidence
        # conf_f1 = []
        # conf_f2 = []

        # for idx, i in enumerate(x):
        #     t = idx + 1
        #     ry_exp = ConfidenceMeasure('renyi')
        #     ts_exp = ConfidenceMeasure('tsallis')
        #     f1 = ry_exp(i, v, t).item()
        #     f2 = ts_exp(i, v, t).item()
        #     print(idx, '\t {:.4f} \t {:.4f}'.format(f1, f2))
        #     conf_f1.append(f1)
        #     conf_f2.append(f2)

        # # remove inf
        # conf_f1 = [i for i in conf_f1 if i != float('inf')]
        # conf_f2 = [i for i in conf_f2 if i != float('inf')]


        # ## divide line
        # print('----------------------------------------')
        # print('mean:\t {:.4f} \t {:.4e}'.format(np.mean(conf_f1), np.mean(conf_f2)))
        # print('max:\t {:.4f} \t {:.4e}'.format(np.max(conf_f1), np.max(conf_f2)))
        # print('min:\t {:.4f} \t {:.4e}'.format(np.min(conf_f1), np.min(conf_f2)))
        # print('std:\t {:.4f} \t {:.4e}'.format(np.std(conf_f1), np.std(conf_f2)))


if __name__ == '__main__':
    unittest.main()