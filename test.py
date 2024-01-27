import os, math, glob, random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from utils.utils import *
from model.model_fast import *

# using cpu mode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Add every font at the specified location
font_dir = ['font']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)

plt.rcParams['font.family'] = 'Happiness Sans'

val_data_path = "data/val"

ckpt_ = glob.glob(os.path.join('checkpoints/*/', '*.h5'))
ckpt_.sort(key=lambda x: os.path.getmtime(x))
ckpt_path = ckpt_[-1]

# ckpt_path = r"checkpoints\backup\model.h5"

# set train params
num_class = len(load_dict()) + 1
#############################################
time_steps = 16
img_shape = (64, 128, 1)
batch_size = 1
#############################################
# random seed is 2023
seed = 2023
random.seed = seed
np.random.seed = seed
tf.random.set_seed(seed)
# set model
model = TinyLPR(
    time_steps=time_steps,
    n_class=num_class,
    n_feat=96,
    width_multiplier=1.0,
    train=True,
).build(img_shape)
model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)


def get_confidence(y_pred):
    _argmax = np.argmax(y_pred[0], axis=-1)
    _idx = _argmax != y_pred.shape[-1] - 1
    confidence = y_pred[0][_idx, _argmax[_idx]]
    print('confidence:', [round(i, 2) for i in confidence])
    confidence = np.min(confidence)
    return confidence


def ctc_decode_fn(y_pred):
    """
    Decodes CTC predictions using greedy search.
    Args:
        y_pred: (batch_size, time_steps, n_class)
    Returns:
        decoded_labels: (batch_size, max_label_len)
    """
    decoded_labels = tf.keras.backend.ctc_decode(
        y_pred=y_pred,
        input_length=tf.ones(shape=(y_pred.shape[0],)) * y_pred.shape[1],
        greedy=True,
    )[0][0]

    return decoded_labels

# predict
def predict():
    img_list = glob.glob(os.path.join(val_data_path, '*.jpg'))
    # shuffle
    random.shuffle(img_list)
    for path in img_list:
        img = Image.open(path).convert('L')
        img = np.array(img)
        # center fit
        img = center_fit(img, img_shape[1], img_shape[0], top_left=True)
        x = np.expand_dims(np.expand_dims(img, axis=-1), axis=0)
        # predict
        y_pred = model.predict(x)
        y_mask, _, y_ctc = y_pred
        y = ctc_decode_fn(y_ctc)
        print('pred:', y)

        _y = decode_label(y_ctc, load_dict())
        print('label:', _y)

        y_conf = get_confidence(y_ctc)
        print('confidence:', y_conf)

        # show mask img
        mask_len = y_mask.shape[-1]
        fig = plt.figure(figsize=(10, 2))
        # less padding
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for idx, i in enumerate(range(10)):
            # keep 2 rows, 10 cols
            ax = fig.add_subplot(2, 10, idx+1)
            ax.axis('off')
            if idx == 0:
                # show origin img
                ax.imshow(img, cmap='gray')
                ax.set_title('pred: {}'.format(_y))
            else:
                # mask * img
                ax.imshow(y_mask[0, :, :, idx-1] * img)

        # save mask img
        plt.savefig('mask.png')
        break


if __name__ == '__main__':
    predict()
    print('ckpt_path:', ckpt_path)
