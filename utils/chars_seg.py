import os
import glob
import random

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# add module path to sys.path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *


# save as png, given opencv img
def save2png(img, save_path):
    # whatever the save path end with any extension, save as png
    save_file, _ = os.path.splitext(save_path)
    # using PIL to save it
    Image.fromarray(img).save(save_file + '.png')


# func to save img as jpg if not jpg
def convert2jpg(img_path):
    if os.path.splitext(img_path)[-1] != '.jpg':
        img = Image.open(img_path)
        img.save(os.path.splitext(img_path)[0] + '.jpg')
        os.remove(img_path)


# function yolo format to x1y1x2y2
def yolo2xyxy(boxes, img_shape, scale=1.05):
    # boxes: [label, x, y, w, h] -> [label, x1, y1, x2, y2]
    boxes = np.array(boxes)
    boxes[:, 1] = boxes[:, 1] * img_shape[1] - boxes[:, 3] * scale * img_shape[1] / 2
    boxes[:, 2] = boxes[:, 2] * img_shape[0] - boxes[:, 4] * scale * img_shape[0] / 2
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3] * scale * img_shape[1]
    boxes[:, 4] = boxes[:, 2] + boxes[:, 4] * scale * img_shape[0]
    return boxes.astype(np.int32)


# function to kmens to find the threshold
def find_threshold(img, k=2):
    # if not gray, convert to gray
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kmeans to find the threshold
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(img.reshape(-1, 1))
    centers = kmeans.cluster_centers_
    centers = np.sort(centers, axis=0)
    threshold = (centers[0] + centers[1]) / 2
    img = cv2.threshold(img, int(threshold), 255, cv2.THRESH_BINARY_INV)[1]
    return img


# function to load img with yolo format txt
def load_img(img_path, txt_path):
    img = cv2_imread(img_path)
    img_shape = img.shape[:2]
    boxes = np.loadtxt(txt_path).reshape(-1, 5)
    boxes = yolo2xyxy(boxes, img_shape)
    return img, boxes


def get_segmask(img_path):
    # get txt path
    txt_path = os.path.join(os.path.splitext(img_path)[0] + '.txt')

    if not os.path.exists(img_path):
        raise FileNotFoundError('img file not found')

    if not os.path.exists(txt_path):
        raise FileNotFoundError('txt file not found')

    img, boxes = load_img(img_path, txt_path)
    # new a same size img fill with 127, channel=1
    seg_img = np.full(img.shape[:2], 127, dtype=np.uint8)

    t_black_pixels = 0
    t_white_pixels = 0
    for box in boxes:
        x1, y1, x2, y2 = box[1:]
        # get cut img
        cut_img = find_threshold(img[y1:y2, x1:x2])
        # count black pixels and white pixels
        black_pixels = np.sum(cut_img == 0)
        white_pixels = np.sum(cut_img == 255)

        t_black_pixels += black_pixels
        t_white_pixels += white_pixels

        seg_img[y1:y2, x1:x2] = cut_img

    # bit_not if white pixels more than black pixels
    if t_white_pixels > t_black_pixels:
        seg_img = cv2.bitwise_not(seg_img)

    # where not 0 or 255, set to 0
    seg_img[seg_img != 0 | 255] = 0

    for box in boxes:
        x1, y1, x2, y2 = box[1:]
        label = int(box[0]) + 1
        seg_img[y1:y2, x1:x2] = seg_img[y1:y2, x1:x2]//255 * label

    return seg_img


if __name__ == '__main__':
    # # convert to jpg if not jpg under given folder
    # for img_path in glob.glob('labeled/*.png'):
    #     convert2jpg(img_path)

    for img_path in glob.glob('data/*.jpg'):
        seg_img = get_segmask(img_path)
        show(seg_img)
        break
        # save2png(seg_img, img_path)

    # img = cv2_imread(img_path)
    # # convert to rgb
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # # show original img and seg img via matplotlib
    # # plt.subplot(121)
    # # plt.imshow(img)
    # # plt.subplot(122)
    # # plt.imshow(seg_img)
    # # plt.show()
