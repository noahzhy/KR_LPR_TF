import os
import glob
import random

from PIL import Image, ImageDraw
from tqdm import tqdm

import sys
sys.path.append('./utils')
from utils import *


def parse_label(path):
    # split label with _
    labels = os.path.splitext(os.path.basename(path))[0].split('_')
    t_labels = ""

    while len(t_labels) < 7:
        label = labels.pop(0)
        # remove space and _
        label = label.replace(' ', '').replace('_', '')
        t_labels += label
        
    len_ = len(t_labels)

    # if A-Z in label, len_ += 1
    if any([c.isupper() for c in t_labels]):
        len_ += 1

    return len_


# count .txt files lines
def count_lines(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # remove empty lines
        lines = [line.strip() for line in lines if line.strip()]
    return len(lines)


# move file to dir
def move2dir(file_path, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # get file name
    file_name = os.path.basename(file_path)
    # get file new path
    new_path = os.path.join(dir_path, file_name)
    # move file, if file exists, overwrite it
    if os.path.exists(new_path):
        os.remove(new_path)
    os.rename(file_path, new_path)


# main
if __name__ == "__main__":
    txts = "E:/dataset/license_plate_chars/checked/*.txt"
    ts = glob.glob(txts)
    # random shuffle
    random.shuffle(ts)
    # # random pick one
    # txt_path = random.choice(glob.glob(txts))
    # label = parse_label(txt_path)
    # print(txt_path, label)

    count = 0
    for txt_path in tqdm(ts):
        # move file if file is exist
        img_path = txt_path.replace('.txt', '.jpg')
        # move2dir(txt_path, 'E:/dataset/license_plate_chars/checked')

        # if img not exists, remove txt
        if not os.path.exists(img_path):
            os.remove(txt_path)
            continue

        # label_size = count_lines(txt_path)
        # label = parse_label(txt_path)
        # len_ = len(label)

        # # if A-Z in label, len_ += 1
        # if any([c.isupper() for c in label]):
        #     len_ += 1

        # print(txt_path, label_size, len_, label)
        # break
    
        # print(txt_path, label_size, len_)
        # if label_size != len_:
        #     # print(txt_path, label_size, len_, label)
        #     print("label_size:", label_size, "\ttrue_len:", len_, "\ttrue_label:", label, "\tfile_path:", txt_path)

            # # get same name img path and xml path
            # img_path = txt_path.replace('.txt', '.jpg')
            # xml_path = txt_path.replace('.txt', '.xml')
            # # move to double_check dir
            # # check if exists
            # if os.path.exists(img_path):
            #     move2dir(img_path, 'E:/dataset/license_plate_chars/double_check')
            # if os.path.exists(xml_path):
            #     move2dir(xml_path, 'E:/dataset/license_plate_chars/double_check')
            # # remove txt
            # os.remove(txt_path)

            # 
            # break
            # count += 1
        # break

    print('count:', count)
