import os, re, sys, glob, time, random

import cv2
import numpy as np
import tensorflow as tf

sys.path.append('./')
from utils.utils import *

# cpu mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def is_valid_label(label: list):
    # list to str
    label = ''.join(label)
    # remove space
    label = label.replace(' ', '')
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


class TFliteDemo:
    def __init__(self, model_path, blank=85, conf_mode="min"):
        self.blank = blank
        self.conf_mode = conf_mode
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def inference(self, x):
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def preprocess(self, img_path):
        image = cv2_imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = center_fit(image, 128, 64, top_left=True)
        image = np.reshape(image, (1, *image.shape, 1)).astype(np.uint8)
        return image

    def get_confidence(self, pred, mode="min"):
        conf = []
        idxs = np.argmax(pred, axis=-1)
        values = np.max(pred, axis=-1)

        for idx, c in zip(idxs, values):
            if idx == self.blank: continue
            conf.append(c/255)

        if mode == "min":
            return np.min(conf)

        return np.mean(conf)

    def postprocess(self, pred):
        label = decode_label(pred, load_dict())
        is_valid = is_valid_label(label)
        conf = self.get_confidence(pred[0], mode=self.conf_mode)
        # keep 4 decimal places
        conf = float('{:.4f}'.format(conf))
        return {
            'label': label,
            'valid': is_valid,
            'confidence': conf,
        }


# main
if __name__ == '__main__':
    num_samples = 20
    # init and load model
    demo = TFliteDemo('save/tiny_lpr_uint8.tflite')

    # get random image
    val_path = "data/val"
    img_list = glob.glob(os.path.join(val_path, '*.jpg'))
    # random sample
    random.shuffle(img_list)
    img_list = img_list[:num_samples]

    res_confs = []

    # warm up for 50 times
    for i in range(50):
        image = demo.preprocess(random.choice(img_list))
        pred = demo.inference(image)

    avg_time = []
    for i in range(len(img_list)):
        image = demo.preprocess(img_list[i])
        # inference
        start = time.process_time()
        pred = demo.inference(image)
        end = time.process_time()
        avg_time.append((end - start) * 1000)
        # post process
        result = demo.postprocess(pred)
        res_confs.append(result)

    # sort by confidence
    res_confs.sort(key=lambda x: x['confidence'], reverse=True)

    for result in res_confs:
        # print dict in format string
        print('label: {label} \tvalid: {valid} \tconfidence: {confidence}'.format(**result))

    print('avg time: {:.4f} ms'.format(np.mean(avg_time)))
