import os, glob, random, math, time, sys

import numba as nb
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import *

# add module path to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *
from utils.order_seg import get_psdeuo_label


# Set loky max cpu count
os.environ['LOKY_MAX_CPU_COUNT'] = '64'

label_dict = load_dict('data/label.names')
raw_label_dict = load_dict('data/label_raw.names')


def shift_img_mask(img, mask, max_len=16):
    # get raw img img_shape
    h, w = img.shape[:2]
    # keep ratio to resize img to 128x64
    ratio = min(128 / w, 64 / h)
    # get resized img_shape
    resize_w, resize_h = int(w * ratio), int(h * ratio)
    # resize img and to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)

    # get shift range
    shift_range_h = 64 - resize_h
    shift_range_w = 128 - resize_w
    
    shift_h = 0 if shift_range_h <= 0 else np.random.randint(0, shift_range_h)
    shift_w = 0 if shift_range_w <= 0 else np.random.randint(0, shift_range_w)

    # shift img
    img_new = np.zeros((64, 128), dtype=np.float32)
    img_new[shift_h:shift_h+resize_h, shift_w:shift_w+resize_w,] = img
    # shift mask
    mask_new = np.zeros((64, 128, max_len), dtype=np.float32)
    mask_new[shift_h:, shift_w:, :] = mask[:64-shift_h, :128-shift_w, :]
    # save img
    return img_new, mask_new


# func to rebuild label
def rebuild_label(label:np.ndarray, blank_index=67):
    # get first two digits
    if label[1] >= 10:
        _raw_label = raw_label_dict[label[0]] + raw_label_dict[label[1]]
        # get dict key via value
        _idx = list(label_dict.values()).index(_raw_label)
        label[0] = _idx
        label[1] = label[2]
    return label


# align label
@nb.jit(nopython=True)
def align_label(label, label_len=8, blank_index=67):
    # align label
    align_label = np.full(label_len, blank_index-1, dtype=np.int32)
    # find first number
    first_num_idx = 0
    for idx, num in enumerate(label):
        if num < 10:
            first_num_idx = idx
            break

    # move label to align_label, last label at position 8
    label_city = label[:first_num_idx]
    label_num = label[first_num_idx:]
    align_label[8-len(label_num):8] = label_num
    align_label[:len(label_city)] = label_city
    return align_label


@nb.jit(nopython=True)
def align_label_ctc(align_label, blank_index=67):
    space = blank_index - 1
    return np.array([
        align_label[0], space, align_label[1], space,
        # align_label[2], align_label[3], align_label[4], space,
        align_label[2], space, align_label[3], space, align_label[4], space,
        align_label[5], space, align_label[6], space, align_label[7]
    ])


# get npy file via given img_path
@nb.jit(nopython=True)
def rect_mask(arr, label=None, max_len=16, fill_value=255):
    out = np.zeros((64, 128, max_len), dtype=arr.dtype)
    empty = np.zeros((64, 128, max_len), dtype=arr.dtype)
    ### realign mask
    realign_mask = np.zeros((64, 128, max_len), dtype=arr.dtype)
    # find last mask
    last_mask_idx = 0
    num_start_idx = 0

    for idx, i in enumerate(label):
        if i < 10:
            num_start_idx = idx
            break

    for idx in range(arr.shape[-1]):
        # check mask is not all zero
        if np.sum(arr[:, :, idx]) == 0: continue
        last_mask_idx = idx

    # move mask to last mask idx
    out[:, :, 8-(last_mask_idx - num_start_idx)-1:8] = arr[:, :, num_start_idx:last_mask_idx+1]

    if label[0] >= 10:
        out[:, :, 0] = arr[:, :, 0]

    # label is raw label
    if label[1] >= 10:
        # sum to one channel
        out[:, :, 0] = np.sum(arr[:, :, :2], axis=-1)

    height, width = out.shape[:2]
    for idx, i in enumerate(range(out.shape[-1])):
        # get mask
        mask = out[:, :, idx]
        # check mask is not all zero
        if np.sum(mask) == 0: continue
        # get mask bbox via np.where
        x, y = np.where(mask > 0)
        min_x, max_x = np.min(x), np.max(x) + 1
        min_y, max_y = np.min(y), np.max(y) + 1

        # c_x = (min_x + max_x) // 2
        # c_y = (min_y + max_y) // 2
        # h = (max_y - min_y) * 1.2
        # w = (max_x - min_x) * 1.2
        # min_x = int(c_x - w // 2)
        # max_x = int(c_x + w // 2)
        # min_y = int(c_y - h // 2)
        # max_y = int(c_y + h // 2)

        # check bbox is valid
        if min_x < 0:       min_x = 0
        if max_x > height:  max_x = height
        if min_y < 0:       min_y = 0
        if max_y > width:   max_y = width
        # fill mask area
        empty[min_x:max_x, min_y:max_y, idx] = 255

    # realign_mask[:, :, 0] = empty[:, :, 0]
    # realign_mask[:, :, 2] = empty[:, :, 1]
    # realign_mask[:, :, 4] = empty[:, :, 2]
    # for i in [5,6,7]: realign_mask[:, :, i] = empty[:, :, 3]
    # realign_mask[:, :, 8] = empty[:, :, 4]
    # realign_mask[:, :, 10] = empty[:, :, 5]
    # realign_mask[:, :, 12] = empty[:, :, 6]
    # realign_mask[:, :, 14] = empty[:, :, 7]

    # for i in range(max_len//2):
    #     realign_mask[:, :, i*2]   = empty[:, :, i]
    #     realign_mask[:, :, i*2+1] = empty[:, :, i]

    for i in [0,1,2,3]: realign_mask[:, :, i] = empty[:, :, 0]
    for i in [4,5]: realign_mask[:, :, i] = empty[:, :, 1]
    for i in [6,7]: realign_mask[:, :, i] = empty[:, :, 2]
    for i in [8,9,10,11]: realign_mask[:, :, i] = empty[:, :, 3]
    for m,n in zip([12,13,14,15], [4,5,6,7]): realign_mask[:, :, m] = empty[:, :, n]

    return realign_mask


class DataLoader(Sequence):
    def __init__(self,
        data_path,
        mode='label', # 'ctc' or 'label
        batch_size=8,
        img_shape=(64, 128, 1),
        time_steps=16,
        num_class=68,
        label_len=8,
        data_augmentation=False,
        shuffle=True):
        super(DataLoader, self).__init__()

        print('Loading data from {}'.format(data_path))
        self.batch_size = batch_size
        self.data_path = data_path
        # dir path or file path
        if os.path.isdir(data_path):
            self.imgs = glob.glob(os.path.join(data_path, '*.*g'))
        elif os.path.isfile(data_path):
            # txt file to list
            with open(data_path, 'r') as f:
                self.imgs = f.readlines()
                self.imgs = [i.strip() for i in self.imgs]
        else:
            raise ValueError('data_path must be dir path or file path')

        self.mode = mode
        assert self.mode in ['ctc', 'label'], 'mode must be ctc or label'

        self.img_shape = img_shape
        self.time_steps = time_steps
        self.num_class = num_class
        self.label_len = label_len
        self.shuffle = shuffle
        self.data_augmentation = data_augmentation
        print('Found {} images'.format(len(self.imgs)))
        self.on_epoch_end()
        # # pick 128 only
        # self.imgs = self.imgs[:1280]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.imgs)

    def __len__(self):
        return int(np.floor(len(self.imgs) / float(self.batch_size)))

    def __getitem__(self, idx):
        h, w = self.img_shape[:2]
        batch_img_path = self.imgs[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_imgs  = np.empty((self.batch_size, h, w, 1),                          dtype=np.float32)
        batch_masks = np.empty((self.batch_size, h, w, self.time_steps),            dtype=np.float32)
        _len = self.time_steps if self.mode == 'ctc' else self.label_len
        batch_labels = np.full((self.batch_size, _len), self.num_class,   dtype=np.int32)

        for idx, img_path in enumerate(batch_img_path):
            if not os.path.exists(img_path):
                print("Not found image: {}".format(img_path))
                continue

            img = cv2_imread(img_path)
            if self.data_augmentation:
                img = self.augmentation(img)

            img_raw = img.copy()
            img = center_fit(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), w, h, inter=cv2.INTER_AREA, top_left=True)

            # load label from txt
            label = np.load(img_path.replace(".jpg", "_label.npy"))

            # load mask from npy
            mask = np.load(img_path.replace(".jpg", ".npy")).astype(np.int32)
            mask = rect_mask(mask, label, max_len=self.time_steps)

            # rebuild label
            label = rebuild_label(label, blank_index=self.num_class)
            # align label
            label = align_label(label, self.label_len, blank_index=self.num_class)
            if self.mode == 'ctc':
                label = align_label_ctc(label, blank_index=self.num_class)

            if self.data_augmentation:
                if np.random.randint(2):
                    img, mask = shift_img_mask(img_raw, mask, max_len=self.time_steps)

            batch_imgs[idx, :, :, 0] = img / 255.
            batch_masks[idx, :, :, :] = mask // 255
            batch_labels[idx, : len(label)] = label.astype(np.int32)

        x = batch_imgs
        y = [batch_masks, batch_labels, batch_labels]
        return x, y

    def augmentation(self, img):
        # negative
        def negative(img):
            return 255 - img

        def random_hsv(img, hue=0.1, sat=1.5, val=1.5):
            hue = np.random.uniform(-hue, hue)
            sat = np.random.uniform(1, sat) if np.random.randint(2) else 1 / np.random.uniform(1, sat)
            val = np.random.uniform(1, val) if np.random.randint(2) else 1 / np.random.uniform(1, val)

            x = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            x = x.astype(np.float32)
            x[:, :, 1] *= sat
            x[:, :, 2] *= val
            x[:, :, 0] += hue * 360
            x[x > 255] = 255
            x[:, :, 1:][x[:, :, 1:] > 255] = 255
            x = x.astype(np.uint8)
            x = cv2.cvtColor(x, cv2.COLOR_HSV2BGR)
            return x

        random_list = [
            # random brightness
            lambda img: random_brightness(img, [0.5, 1.5]),
            # random hsv
            lambda img: random_hsv(img, hue=0.5, sat=2, val=2),
            # random negative
            lambda img: negative(img),
        ]
        # apply random one of the above transform
        img = random.choice(random_list)(img)
        return img


# main
if __name__ == '__main__':
    # mode = 'label'
    mode = 'ctc'
    time_steps = 16
    label_len = 8
    data_path = "/home/noah/datasets/val"
    # data_path = "train.txt"
    loader = DataLoader(
        data_path,
        mode=mode,
        batch_size=8,
        time_steps=time_steps,
        label_len=label_len,
        data_augmentation=True,
        num_class=len(load_dict('data/label.names')),
    )

    for x, y in loader:
        print(type(x), type(y))
        print(x[0].shape, y[0].shape, y[1].shape)
        print(y[1])
        # print(np.unique(y[0]))
        break

    plt.figure(figsize=(21, 1))
    plt.tight_layout()

    for imgs, y in loader:
        masks, _, labels = y
        _label_len = labels.shape[-1]
        # | img | mask1 | mask2 | ... | mask N |
        plt.subplot(1, masks.shape[-1]+1, 1)
        plt.imshow(imgs[0, :, :, 0], cmap='gray')
        plt.axis('off')

        for i in range(masks.shape[-1]):
            plt.subplot(1, time_steps+1, i+2)
            plt.imshow(masks[0, :, :, i], cmap='gray')
            if i < _label_len:
                plt.title(labels[0, i])
            plt.axis('off')

        # plt.show()
        # save
        plt.savefig('data.png', bbox_inches='tight')
        break
