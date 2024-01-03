import os
import glob
import random

import cv2
import numpy as np
from PIL import Image, ImageFont
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from tqdm import tqdm

# from .utils import *
from utils import *


def show_img(img):
    plt.imshow(img)
    plt.show()


def sort_coordinates(coords, txt_path):
    # [labels, center_x, center_y]
    # count labels which is bigger than 9
    # count = np.sum(coords[:, 0] > 9)
    # if space in labels
    if txt_path.find(' ') != -1:
        # divide coordinates into two parts via center_y
        row1 = coords[coords[:, 2] < np.median(coords[:, 2])]
        row2 = coords[coords[:, 2] >= np.median(coords[:, 2])]
        # sort coordinates_1 via center_x
        row1 = row1[row1[:, 1].argsort()]
        # sort coordinates_2 via center_x
        row2 = row2[row2[:, 1].argsort()]
        # concatenate coordinates_1 and coordinates_2
        coords = np.concatenate([row1, row2], axis=0)
    else:
        # sort coordinates via center_x
        coords = coords[coords[:, 1].argsort()]
    return coords


# function yolo format to x1y1x2y2
def yolo2xyxy(boxes, img_shape, scale=1.05):
    # boxes: [label, x, y, w, h] -> [label, x1, y1, x2, y2]
    boxes = np.array(boxes)
    boxes[:, 1] = boxes[:, 1] * img_shape[1] - boxes[:, 3] * scale * img_shape[1] / 2
    boxes[:, 2] = boxes[:, 2] * img_shape[0] - boxes[:, 4] * scale * img_shape[0] / 2
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3] * scale * img_shape[1]
    boxes[:, 4] = boxes[:, 2] + boxes[:, 4] * scale * img_shape[0]
    # center point
    center_point_x = (boxes[:, 1] + boxes[:, 3]) / 2
    center_point_y = (boxes[:, 2] + boxes[:, 4]) / 2
    # label
    labels = boxes[:, 0].astype(np.int32)
    # concat label, center_x, center_y, box to [label, center_x, center_y, box]
    box = np.concatenate([center_point_x.reshape(-1, 1), center_point_y.reshape(-1, 1), boxes[:, 1:]], axis=1)
    labels = labels.reshape(-1, 1)
    return np.concatenate([labels, box], axis=1).astype(np.int32)


# function to kmens to find the threshold
def find_threshold(img, k=2):
    # img should be gray channel
    if len(img.shape) != 2:
        raise Exception('img should be gray channel')
    # kmeans to find the threshold
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(img.reshape(-1, 1))
    centers = kmeans.cluster_centers_

    if len(centers) == 1:
        return img

    centers = np.sort(centers, axis=0)
    threshold = (centers[0] + centers[1]) / 2
    img = cv2.threshold(img, int(threshold), 255, cv2.THRESH_BINARY_INV)[1]
    return img


def get_mask(img, boxes):
    # new a same size img fill with 127, channel=1
    seg_img = np.full(img.shape[:2], 127, dtype=np.uint8)

    black_pxs = 0
    white_pxs = 0
    for box in boxes:
        x1, y1, x2, y2 = box
        # check if out of range
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > img.shape[1]: x2 = img.shape[1]
        if y2 > img.shape[0]: y2 = img.shape[0]
        
        # check if x1 == x2 or y1 == y2
        if x1 == x2 or y1 == y2:
            return np.zeros(img.shape[:2], dtype=np.uint8)

        # get cut img
        if len(img.shape) == 3:
            # convert to gray
            cut_img = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        else:
            cut_img = img[y1:y2, x1:x2]

        # find threshold
        cut_img = find_threshold(cut_img)
        # count black pixels and white pixels
        b_px = np.sum(cut_img == 0)
        w_px = np.sum(cut_img == 255)
        black_pxs += b_px
        white_pxs += w_px
        seg_img[y1:y2, x1:x2] = cut_img

    # bit_not if white pixels more than black pixels
    if white_pxs > black_pxs:
        seg_img = cv2.bitwise_not(seg_img)

    # where not 0 or 255, set to 0
    seg_img[seg_img != 0 | 255] = 0
    return seg_img


def get_centers(img_path, txt_path, scale=1.05):
    img = cv2_imread(img_path)
    img_shape = img.shape[:2]
    boxes = np.genfromtxt(txt_path, delimiter=' ', dtype=np.float32).reshape(-1, 5)
    boxes = yolo2xyxy(boxes, img_shape)
    return img, sort_coordinates(boxes, txt_path)


def get_order_mask(img_path, txt_path):
    img, boxes = get_centers(img_path, txt_path)
    labels = boxes[:, 0]
    boxes = boxes[:, 3:]
    img = get_mask(img, boxes)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        img[y1:y2, x1:x2] = img[y1:y2, x1:x2]//255 * (idx+1)

    return img, labels


# save as png, given opencv img
def save2png(img, save_path):
    # whatever the save path end with any extension, save as png
    save_file, _ = os.path.splitext(save_path)
    Image.fromarray(img).save(save_file + '.png')


# func to save img as jpg if not jpg
def convert2jpg(img_path):
    if os.path.splitext(img_path)[-1] != '.jpg':
        img = Image.open(img_path)
        img.save(os.path.splitext(img_path)[0] + '.jpg')
        os.remove(img_path)


# func to convert img to [TxHxW] numpy array
def get_psdeuo_label(img_path, h=64, w=128, t=16, top_left=True, inter=cv2.INTER_NEAREST):
    txt_path = img_path.replace(".jpg", ".txt")
    if not os.path.exists(txt_path):
        raise Exception("txt file not exists:", txt_path)

    # print('get pseudo label:', img_path)
    mask, label = get_order_mask(img_path, txt_path)
    # # resize to [h, w]
    # mask = center_fit(mask, w, h, top_left=top_left)
    img = np.zeros((h, w), dtype=np.uint8)
    # convert to [TxHxW] numpy array
    arr = np.zeros((img.shape[0], img.shape[1], t), dtype=np.uint8)
    for i in range(t):
        # img[:mask.shape[0], :mask.shape[1]] = mask
        _mask = mask == i+1
        _mask = _mask.astype(np.uint8) * 255
        _mask = center_fit(_mask, w, h, top_left=top_left, inter=inter)
        arr[:, :, i] = _mask

    return arr, label


if __name__ == '__main__':
    max_len_label = 16
    dict_path = 'data/label.names'
    _dict = load_dict(dict_path)
    print(_dict)

    # # korean unicode scope: 0xAC00 ~ 0xD7AF
    # data_path = "E:/dataset/license_plate_chars/val/*[가-힣]*.jpg"
    # img_path = random.choice(glob.glob(data_path))
    # img_path = 'E:/dataset/license_plate_chars/tmp/57노8464_20230116050217_13219585.jpg'
    # print('img_path:', img_path)
    # arr, label = get_psdeuo_label(img_path, 64, 128, max_len_label, top_left=True)
    # # random select one img

    data_path = "/home/noah/datasets/train/*.txt"
    img_paths = glob.glob(data_path)
    random.shuffle(img_paths)

    for img_path in tqdm(img_paths):
        img_path = img_path.replace('.txt', '.jpg')
        try:
            arr, label = get_psdeuo_label(img_path, 64, 128, max_len_label, top_left=True)
            # print('arr:', np.max(arr), np.min(arr), img_path, label)
            # save arr and label to npy
            np.save(img_path.replace('.jpg', '.npy'), arr)
            np.save(img_path.replace('.jpg', '_label.npy'), label)

        except Exception as e:
            print(e, img_path)

            # num_t = arr.shape[2]
            # fig = plt.figure(figsize=(num_t, 1))
            # plt.tight_layout()
            # for i in range(num_t):

            #     try:
            #         plt.subplot(1, num_t, i+1)
            #         plt.imshow(arr[:, :, i], cmap='gray')
            #         plt.title(_dict[i])
            #         plt.axis('off')
            #     except Exception as e:
            #         print(img_path)

            # plt.show()
            # break
