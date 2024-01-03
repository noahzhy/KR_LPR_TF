import os, sys, glob, math, random, shutil

import numpy as np
import pandas as pd

sys.path.append('./')
from deploy.tflite_demo import TFliteDemo


if __name__ == '__main__':
    # init and load model
    model = TFliteDemo('save/tiny_lpr_uint8.tflite')
    path = "/home/noah/datasets/train"
    img_list = glob.glob(os.path.join(path, '*.jpg'))

    f = open('df.csv', 'w+')
    f.write('path,label,valid,confidence\n')
    for i in range(len(img_list)):
        print(i, img_list[i])
        data = {}

        image = model.preprocess(img_list[i])
        pred = model.inference(image)
        result = model.postprocess(pred)

        f.write('{},{},{},{}\n'.format(img_list[i], result['label'], result['valid'], result['confidence']))
        f.flush()
    
    f.close()
