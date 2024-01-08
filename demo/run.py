import os, glob
from itertools import groupby

import cv2
import numpy as np
import gradio as gr
import tensorflow as tf


def get_sample_images():
    list_ = glob.glob(os.path.join(os.path.dirname(__file__), 'samples/*.jpg'))
    # sort by name
    list_.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return [[i] for i in list_]


def cv2_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def decode_label(mat, chars) -> str:
    # mat is the output of model
    # get char indices along best path
    best_path_indices = np.argmax(mat[0], axis=-1)
    # collapse best path (using itertools.groupby), map to chars, join char list to string
    best_chars_collapsed = [chars[k] for k, _ in groupby(best_path_indices) if k != len(chars)]
    res = ''.join(best_chars_collapsed)
    # remove space and '_'
    res = res.replace(' ', '').replace('_', '')
    return res


def center_fit(img, w, h, inter=cv2.INTER_NEAREST, top_left=True):
    # get img shape
    img_h, img_w = img.shape[:2]
    # get ratio
    ratio = min(w / img_w, h / img_h)

    if len(img.shape) == 3:
        inter = cv2.INTER_AREA
    # resize img
    img = cv2.resize(img, (int(img_w * ratio), int(img_h * ratio)), interpolation=inter)
    # get new img shape
    img_h, img_w = img.shape[:2]
    # get start point
    start_w = (w - img_w) // 2
    start_h = (h - img_h) // 2

    if top_left:
        start_w = 0
        start_h = 0

    if len(img.shape) == 2:
        # create new img
        new_img = np.zeros((h, w), dtype=np.uint8)
        new_img[start_h:start_h+img_h, start_w:start_w+img_w] = img
    else:
        new_img = np.zeros((h, w, 3), dtype=np.uint8)
        new_img[start_h:start_h+img_h, start_w:start_w+img_w, :] = img

    return new_img


def load_dict(dict_path='label.names'):
    with open(dict_path, 'r', encoding='utf-8') as f:
        _dict = f.read().splitlines()
    _dict = {i: _dict[i] for i in range(len(_dict))}
    return _dict


class TFliteDemo:
    def __init__(self, model_path, label_path="label.names", blank=68, conf_mode="mean"):
        self.blank = blank
        self.conf_mode = conf_mode
        self.label_dict = load_dict(label_path)
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inputs = self.interpreter.get_input_details()
        self.outputs = self.interpreter.get_output_details()

    def inference(self, x):
        self.interpreter.set_tensor(self.inputs[0]['index'], x)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.outputs[0]['index'])

    def preprocess(self, img):
        if isinstance(img, str):
            image = cv2_imread(img)
        else:
            # check none
            if img is None:
                raise ValueError('img is None')
            image = img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = center_fit(image, 128, 64, top_left=True)
        image = np.reshape(image, (1, *image.shape, 1)).astype(np.uint8)
        return image

    def get_confidence(self, pred, mode="mean"):
        _argmax = np.argmax(pred, axis=-1)
        _idx = _argmax != pred.shape[-1] - 1
        conf = pred[_idx, _argmax[_idx]] / 255.0
        return np.min(conf) if mode == "min" else np.mean(conf)

    def postprocess(self, pred):
        label = decode_label(pred, self.label_dict)
        conf = self.get_confidence(pred[0], mode=self.conf_mode)
        # keep 4 decimal places
        conf = float('{:.4f}'.format(conf))
        return label, conf

    def run(self, img):
        img = self.preprocess(img)
        pred = self.inference(img)
        return self.postprocess(pred)
