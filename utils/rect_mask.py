import os, sys, glob, random, shutil, math

import numpy as np
from PIL import Image, ImageDraw


# get npy file via given img_path
def get_npy(arr):
    out = arr.copy()

    for idx in range(arr.shape[-1]):
        # get mask
        mask = arr[:, :, idx]

        # check mask is not all zero
        if np.all(mask == 0):
            continue

        # get mask bbox via np.where
        x, y = np.where(mask > 0)
        minx, maxx = np.min(x), np.max(x)+1
        miny, maxy = np.min(y), np.max(y)+1
        # fill mask area
        out[minx:maxx, miny:maxy, idx] = 255
        # # save mask as mask_{idx}.jpg
        # mask = Image.fromarray(out[:, :, idx]).convert('L')
        # mask.save('mask_{}.jpg'.format(idx))

    return out


# main func
if __name__ == "__main__":
    path = "/home/noah/datasets/train"
    img_list = glob.glob(os.path.join(path, '*.jpg'))
    random.shuffle(img_list)
    get_npy(img_list[0])
