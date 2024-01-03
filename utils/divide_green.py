import os
import sys
import glob
from shutil import copyfile

import cv2
import numpy as np
from PIL import Image

from utils import *


def convert2jpg(img_path):
    if os.path.splitext(img_path)[-1] != '.jpg':
        img = Image.open(img_path)
        img.save(os.path.splitext(img_path)[0] + '.jpg')
        os.remove(img_path)


# func to divide green img
def divide_green(img_path):
    # if extension is not jpg, convert to jpg
    img_ext = os.path.splitext(img_path)[-1]
    if img_ext != '.jpg':
        print('convert to jpg:', img_path)
        convert2jpg(img_path)

    # update img path end with jpg
    img_path = os.path.splitext(img_path)[0] + '.jpg'
    img = cv2_imread(img_path)

    # to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # get green mask
    g_mask = cv2.inRange(hsv, (35, 43, 46), (77, 255, 255))
    # get yellow mask
    y_mask = cv2.inRange(hsv, (11, 43, 46), (34, 255, 255))
    # get white mask
    w_mask = cv2.inRange(hsv, (0, 0, 221), (180, 30, 255))
    # black mask
    b_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 46))
    # # gray mask
    # gry_mask = cv2.inRange(hsv, (0, 0, 46), (180, 43, 220))
    # # get blue mask
    # b_mask = cv2.inRange(hsv, (100, 43, 46), (124, 255, 255))

    # if green is max
    if g_mask.sum() > y_mask.sum():
    # if g_mask.sum() > y_mask.sum() and g_mask.sum() > w_mask.sum() and g_mask.sum() > b_mask.sum() and g_mask.sum() > b_mask.sum() and g_mask.sum() > gry_mask.sum():
        print('green is max:', img_path)
        # copy to mv_path
        copyfile(img_path, os.path.join(mv_path, os.path.basename(img_path)))
        # remove img_path
        os.remove(img_path)


# main
if __name__ == "__main__":
    root_path = "E:/dataset/license_plate_chars/tmp/Images"
    mv_path = "E:/dataset/license_plate_chars/gg_tmp"
    # get all img paths which end with jpg or png
    img_paths = glob.glob(os.path.join(root_path, '*.*g'))
    print('img_paths:', len(img_paths))
    # divide green
    for img_path in img_paths:
        divide_green(img_path)
