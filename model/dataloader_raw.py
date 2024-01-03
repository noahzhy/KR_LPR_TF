import os
import glob

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# add module path to sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *
from utils.order_seg import get_psdeuo_label


# Set loky max cpu count
os.environ['LOKY_MAX_CPU_COUNT'] = '4'


class DataLoader(Sequence):
    def __init__(self,
        data_path,
        batch_size=32,
        img_shape=(64, 128),
        char_mask_shape=(64, 128),
        max_len_label=10,
        num_class=85,
        shuffle=True):
        super(DataLoader, self).__init__()
        print('Loading data from {}'.format(data_path))
        self.batch_size = batch_size
        self.data_path = data_path
        self.img_shape = img_shape
        self.char_mask_shape = char_mask_shape
        self.max_len_label = max_len_label
        self.num_class = num_class
        self.shuffle = shuffle
        self.imgs = glob.glob(os.path.join(data_path, '*.*g'))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.imgs) / float(self.batch_size)))
        # return int(np.floor(len(self.imgs) / self.batch_size))

    def __getitem__(self, idx):
        h, w = self.img_shape
        batch_img_path = self.imgs[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_imgs  = np.empty((self.batch_size, *self.img_shape, 1), dtype=np.float32)
        batch_masks = np.empty((self.batch_size, *self.char_mask_shape, self.max_len_label), dtype=np.int32)
        batch_labels= np.full((self.batch_size, self.max_len_label), self.num_class, dtype=np.int32)
        for idx, img_path in enumerate(batch_img_path):
            img = cv2_imread(img_path)
            img = center_fit(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), w, h, inter=cv2.INTER_AREA, top_left=True)
            if img is None:
                print('img is None:', img_path)
                continue
            mask, label = get_psdeuo_label(img_path, *self.char_mask_shape, t=self.max_len_label, top_left=True)

            batch_imgs[idx, :, :, 0] = img / 255.
            batch_masks[idx, :, :, :] = mask // 255
            batch_labels[idx, : len(label)] = label

        x = batch_imgs
        y = [batch_masks, batch_labels]
        return [x, y], y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.imgs)


# main
if __name__ == '__main__':
    data_path = "E:/dataset/license_plate_chars/train"
    loader = DataLoader(data_path, batch_size=64)
    # len 
    print(len(loader))
    # check last batch
    print(len(loader.imgs) % loader.batch_size)
    # check batch

    for x, y in loader:
        print(type(x), type(y))
        print(x[0].shape, x[1][0].shape, x[1][1].shape)
        # break


    # plt.figure(figsize=(10, 1))
    # plt.tight_layout()

    # for x, y in loader:
    #     print(type(x), type(y))
    
    # for imgs, masks, labels in loader:
    #     print(imgs.shape, masks.shape, labels.shape)
    #     # first one is img, horizontal 10 is mask
    #     # | img | mask1 | mask2 | ... | mask10 |
    #     plt.subplot(1, 11, 1)
    #     plt.imshow(imgs[0, :, :, 0], cmap='gray')
    #     plt.axis('off')

    #     for i in range(10):
    #         plt.subplot(1, 11, i+2)
    #         plt.imshow(masks[0, :, :, i], cmap='gray')
    #         plt.title(labels[0, i])
    #         plt.axis('off')

    #     plt.show()

    #     break
