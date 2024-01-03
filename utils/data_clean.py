import os, sys, random, shutil, math, glob, re

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageStat
import matplotlib.pyplot as plt

from utils import move2dir, center_fit


target_path = "/home/noah/datasets/train_bad"

# img size, keep ratio via given width or height, PIL
def keep_ratio(img: Image, width:int=None, height:int=None):
    W, H = img.size
    # ratio
    w_ratio = width / W
    h_ratio = width / H
    ratio = min(w_ratio, h_ratio)
    # new size
    new_size = (int(W * ratio), int(H * ratio))
    # resize
    img = img.resize(new_size)
    return img


# func to merge imgs into one via given img paths
def merge_imgs(img_paths: list, save_path='merge.png'):
    # merge imgs into one
    # via PIL
    # get img shape
    W, H = 64, 128
    # create a new image
    new_img = Image.new('RGB', (W * len(img_paths), H))
    # merge
    for i, img_path in enumerate(img_paths):
        img = Image.open(img_path)
        # size fit
        img = keep_ratio(img, W)
        new_img.paste(img, (i * W, 0))
    # save
    new_img.save(save_path)


def is_valid_label(label: list):
    # list to str
    label = ''.join(label)
    _city = [
        '서울', '부산', '대구', '인천', '광주',
        '대전', '울산', '세종', '경기', '강원',
        '충북', '충남', '전북', '전남', '경북',
        '경남', '제주',
    ]
    _pattern = r'^[가-힣]{2}[0-9]{2}[가-힣]{1}[0-9]{4}|^[0-9]{2,3}[가-힣]{1}[0-9]{4}$'
    # is valid
    if re.match(_pattern, label):
        return label[:2].isdigit() or label[:2] in _city
    else:
        return False


# func to parse label
def parse_label(label:np.ndarray):
    # get last 5 digits
    label = label[-5:]
    if label[0] >= 10:
        for i in label[1:]:
            if i >= 10:
                return False
        return True
    else:
        return False

    return False


# func to list all images in a folder
def list_images(path):
    # image types: jpg, png, jpeg,
    # return a list of image path, via glob
    return glob.glob(os.path.join(path, '*.jpg')) + glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpeg'))


def find_match_npy(img_path):
    # find match npy file
    img_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_name)[0]
    npy_path = os.path.join(os.path.dirname(img_path), img_name + '.npy')
    return npy_path


# func to calculate the Picture lightness
def calculate_brightness(image_path):
    # calculate brightness of the image
    # via PIL
    img = Image.open(image_path).convert('L')
    stat = ImageStat.Stat(img)
    return stat.mean[0]


def caluclate_max_area(npy_path):
    # calculate max area of the npy file
    # shape
    npy = np.load(npy_path)
    H, W, T = npy.shape
    # add all channels
    npy = np.sum(npy, axis=-1)

    # min x and max x
    min_x = np.where(np.sum(npy, axis=0) > 0)[0][0]
    max_x = np.where(np.sum(npy, axis=0) > 0)[0][-1]
    w_ = max_x - min_x
    r_w = w_ / W

    min_y = np.where(np.sum(npy, axis=1) > 0)[0][0]
    max_y = np.where(np.sum(npy, axis=1) > 0)[0][-1]
    h_ = max_y - min_y
    r_h = h_ / H
    r_h_w = r_h / r_w


    # fill ratio
    w_pixel = np.sum(npy > 0)

    # to openCV format
    npy = npy.astype(np.uint8)
    # find minarea rect of the white area via OpenCV
    contours, hierarchy = cv2.findContours(npy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw min area rect
    cnts = []
    for cnt in contours:
        cnts.extend(cnt)

    cnts = np.array(cnts)
    rect = cv2.minAreaRect(cnts)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # find max width and height in box
    w = np.max(box[:, 0]) - np.min(box[:, 0])
    h = np.max(box[:, 1]) - np.min(box[:, 1])
    # ratio
    # _r_w = w / W
    # _r_h = h / H
    r_h_w = h / w

    # draw
    # img = cv2.drawContours(npy, [box], 0, (255, 255, 255), 2)

    # # save as npy.png via PIL
    # img = Image.fromarray(npy.astype(np.uint8))
    # img.save('npy.png')

    # calculate area
    max_area = cv2.contourArea(box)
    all_area = H * W
    ratio = max_area / all_area
    f_ratio = w_pixel / max_area

    return ratio, f_ratio, r_w, r_h, r_h_w


    # if ratio < 0.10:
    #     print('ratio:', ratio, 'f_ratio:', f_ratio)
    #     # quit()
    #     # to one channel image
    #     # save as npy.png via PIL
    #     img = Image.fromarray(npy.astype(np.uint8))
    #     img.save('npy.png')


# find npy file and _label.npy file via given image path
def find_match_npy_label(img_path):
    # find match npy file
    img_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_name)[0]
    npy_path = os.path.join(os.path.dirname(img_path), img_name + '.npy')
    label_path = os.path.join(os.path.dirname(img_path), img_name + '_label.npy')
    return npy_path, label_path


def is_mask_match_label(img_path):
    # find match npy file
    img_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_name)[0]
    npy_path = os.path.join(os.path.dirname(img_path), img_name + '.npy')
    label_path = os.path.join(os.path.dirname(img_path), img_name + '_label.npy')
    
    count = 0
    # num of mask which is not 0
    mask = np.load(npy_path).astype(np.uint8)
    for i in range(mask.shape[-1]):
        _mask = mask[:, :, i]
        mask_num = np.sum(_mask > 0)
        if mask_num > 16:
            count += 1
            
    correct_label = np.load(label_path)
    if count != len(correct_label):
        # print("mask num:", count, "correct label:", correct_label)
        return False

    return True


# func to calculate the Picture clarity
def calcClarity(img):
    raw = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # normalize to min max
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    std = np.std(img)
    # 高斯模糊
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Sobel算子
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # 分别计算x，y方向上的梯度平方
    sobelx = np.uint64(sobelx)
    sobely = np.uint64(sobely)
    sobelx2 = np.uint64(sobelx * sobelx)
    sobely2 = np.uint64(sobely * sobely)
    # 对梯度平方和进行开方
    sobelxy = np.uint64(np.sqrt(sobelx2 + sobely2))
    # 计算梯度平方和的均值
    mean = np.mean(sobelxy)
    return mean, std


def num2char(num_list: list, dict_path: str, blank_index: int = -1):
    """
    Convert number to character
    :param num_list: list of number
    :param dict_path: path to dict file
    :return: list of character
    """
    # dict load
    with open(dict_path, 'r', encoding='utf-8') as f:
        char_dict = f.read().splitlines()

    char_dict = {i: char_dict[i] for i in range(len(char_dict))}
    char_list = [char for char in [char_dict[num]
                                   for num in num_list if num != blank_index]]

    return char_list


# main
if __name__ == '__main__':
    # list all images
    data_path = "/home/noah/datasets/train"
    # data_path = "/home/noah/datasets/val"
    img_list = list_images(data_path)
    print("total images:", len(img_list))

    # quit()
    debug = True

    # random pick some images
    if debug:
        num_pick = 2000
        img_list = random.sample(img_list, num_pick)
    else:
        img_list = tqdm(img_list)

    dis_img = []
    # calc clarity
    img_info = {}
    for img_path in img_list:
        img = cv2.imread(img_path)
        img = center_fit(img, 128, 64, top_left=True)
        # lightness = calculate_brightness(img_path)
        r_f, f_ratio, r_w, r_h, r_h_w = caluclate_max_area(find_match_npy(img_path))

        if r_w < 0.7 and r_h_w < 0.4:
            if debug:
                dis_img.append(img_path)
            else:
                move2dir(img_path, target_path)
                npy_f, label_f = find_match_npy_label(img_path)
                move2dir(npy_f, target_path)
                move2dir(label_f, target_path)

            # img_list.set_description("move: {} to {}".format(img_path, target_path))

            # print('move: {} to {}'.format(img_path, target_path))

        # npy_f, label_f = find_match_npy_label(img_path)
        # label = np.load(label_f)
        # # to list
        # label = label.tolist()
        # status = is_valid_label(num2char(label, 'data/label_raw.names'))

        # status = parse_label(label)
        # if status:
        #     continue
        # else:
        #     print("label:", label, img_path)

        if debug:
            clarity, std = calcClarity(img)
            img_info[img_path] = {
                'clarity': clarity,
                'std': std,
                'area_HW': r_f,
                'pixel_area': f_ratio,
                'r_w': r_w,
                'r_h': r_h,
                'r_h_w': r_h_w,
                'pixel_HW': r_f,
            }

    # # save as one
    if debug:
        merge_imgs(dis_img, 'merge.png')
        fig = plt.figure(figsize=(10, 10))
        # img_info to dataframe
        df = pd.DataFrame(img_info)
        df = df.T

        x_pick = df['r_w']
        y_pick = df['r_h_w']
        # add subplot
        ax1 = fig.add_subplot(313)
        ax1.scatter(x_pick, y_pick, marker='o', color='b')
        # get min value in x
        min_x = x_pick.min()
        print('min_x:', min_x)
        # get key
        min_path = df[x_pick == min_x].index[0]
        print(df[x_pick == min_x])
        # add lowest clarity image to plot via PIL
        img = Image.open(min_path)
        npy_f, label_f = find_match_npy_label(min_path)
        npy = np.load(npy_f)
        npy = np.sum(npy, axis=-1)

        img = np.array(img)
        img = center_fit(img, 128, 64, top_left=True)
        ax2 = fig.add_subplot(321)
        ax2.set_title('lowest clarity')
        ax2.imshow(img)
        img = Image.fromarray(npy.astype(np.uint8))
        ax4 = fig.add_subplot(323)
        ax4.set_title(np.load(label_f))
        ax4.imshow(img)

        x_max = x_pick.max()
        # print('max_x:', x_max)
        # get key
        max_path = df[x_pick == x_max].index[0]
        print(df[x_pick == x_max])

        # add highest clarity image to plot via PIL
        img = Image.open(max_path)
        # show it's label
        npy_f, label_f = find_match_npy_label(max_path)
        img = np.array(img)
        img = center_fit(img, 128, 64, top_left=True)
        ax3 = fig.add_subplot(322)
        ax3.set_title('highest clarity')
        ax3.imshow(img)
        npy = np.load(npy_f)
        npy = np.sum(npy, axis=-1)
        img = Image.fromarray(npy.astype(np.uint8))
        ax5 = fig.add_subplot(324)
        ax5.set_title(np.load(label_f))
        ax5.imshow(img)

        plt.tight_layout()
        # save
        plt.savefig('clarity.png')
        print('save to clarity.png')
