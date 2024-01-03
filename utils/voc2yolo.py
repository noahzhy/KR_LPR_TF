import os
import glob
import random

import cv2
import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ET
import xml.etree.ElementTree as ET


# load txt file as dict
def load_dict(txt_path):
    _dict = {}
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            _dict[line] = len(_dict)

    return _dict


# convert dict
dict_chars = load_dict('E:/dataset/license_plate_chars/label.names')
print(dict_chars)


# func to convert voc to yolo
def voc2yolo(voc_path):
    # read xml
    tree = ET.parse(voc_path)
    root = tree.getroot()
    # get img shape from voc format
    img_h = float(root.find('size').find('height').text)
    img_w = float(root.find('size').find('width').text)
    # print(img_h, img_w)
    # get objects
    objects = root.findall('object')
    # get boxes
    boxes = []
    for obj in objects:
        # get label
        label = obj.find('name').text
        # convert to index
        label = int(dict_chars[label])
        # get bbox
        bbox = obj.find('bndbox')
        # get x1, y1, x2, y2
        x1 = float(bbox.find('xmin').text) / img_w
        y1 = float(bbox.find('ymin').text) / img_h
        x2 = float(bbox.find('xmax').text) / img_w
        y2 = float(bbox.find('ymax').text) / img_h
        # convert to yolo format x, y, w, h
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        # append to boxes
        boxes.append([label, x, y, w, h])
        # # append to boxes
        # boxes.append([label, x1, y1, x2, y2])

    # change extension to txt
    yolo_path = os.path.splitext(voc_path)[0] + '.txt'

    # save to yolo_path
    with open(yolo_path, 'w+') as f:
        for box in boxes:
            # to all float
            box = [str(i) for i in box]
            # join with space
            box = ','.join(box)
            # write to file
            f.write(box + '\n')
    # print
    # print('convert to yolo format:', xml)


# main
if __name__ == "__main__":
    voc_dir = "E:/dataset/license_plate_chars/tmp/Images"
    vocs = glob.glob(os.path.join(voc_dir, '*.xml'))
    for voc in tqdm.tqdm(vocs):
        voc2yolo(voc)
