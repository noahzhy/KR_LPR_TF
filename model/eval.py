import os, re, sys, glob, math, random

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import load_dict


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


def ctc_decode_fn(y_pred):
    """
    Decodes CTC predictions using greedy search.
    Args:
        y_pred: (batch_size, time_steps, n_class)
    Returns:
        decoded_labels: (batch_size, max_label_len)
    """
    decoded_labels = tf.keras.backend.ctc_decode(
        y_pred=y_pred, input_length=tf.ones(shape=(y_pred.shape[0],)) * y_pred.shape[1], greedy=True
    )[0][0]

    return decoded_labels


class CTCAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, log_dir, blank_index=86, mode='train', name='ctc_acc'):
        super(CTCAccuracyCallback, self).__init__()
        self.validation_data = validation_data
        self.ctc_decode_fn = ctc_decode_fn
        self.log_dir = log_dir
        self.blank_index = blank_index
        self.mode = mode

        # save best
        self.best_ctc_acc = 0.0
        self.best_char_acc = 0.0

        # writer
        self.writer = tf.summary.create_file_writer(
            os.path.join(log_dir, name))

    def on_epoch_end(self, epoch, logs=None):
        # Extract the validation data and labels
        val_data = self.validation_data
        size = len(val_data)
        t_correct_ctc = 0
        t_correct_char = 0
        t_RE = 0
        t_num = 0
        t_kor = 0

        for i in range(len(val_data)):
            x_val, y_val = val_data[i]
            y_ctc = y_val[-1]

            # Use the CTC decoding function to get predicted labels
            y_pred = self.model.predict(x_val, verbose=0)[-1]
            decoded_labels = self.ctc_decode_fn(y_pred=y_pred)

            ctc_acc, char_acc, ctc_acc_RE, e_num, e_kor = self.calc_ctc_acc(decoded_labels, y_ctc)
            t_correct_ctc += ctc_acc
            t_correct_char += char_acc
            t_RE += ctc_acc_RE

            t_num += e_num
            t_kor += e_kor

        t_ctc_acc = t_correct_ctc / size
        t_char_acc = t_correct_char / size
        t_RE = t_RE / size

        t_num = t_num / size
        t_kor = t_kor / size

        print("RE   acc.: {:.2f} %".format(t_RE * 100))
        print("CTC  acc.: {:.2f} %".format(t_ctc_acc * 100))
        print("Char acc.: {:.2f} %".format(t_char_acc * 100))
        print("Num  err.: {:.2f} %".format(t_num * 100))
        print("Kor  err.: {:.2f} %".format(t_kor * 100))

        if self.mode == 'val':
            return

        # update best
        if t_ctc_acc >= self.best_ctc_acc:
            self.best_ctc_acc = t_ctc_acc
            self.model.save(
                "checkpoints/ctc_{:.4f}_char_{:.4f}.h5".format(t_ctc_acc, t_char_acc))

        if t_char_acc >= self.best_char_acc:
            self.best_char_acc = t_char_acc
            self.model.save(
                "checkpoints/ctc_{:.4f}_char_{:.4f}.h5".format(t_ctc_acc, t_char_acc))

        # Save logs to TensorBoard
        if self.log_dir:
            with self.writer.as_default():
                tf.summary.scalar("RE   accuracy",  t_RE,       step=epoch)
                tf.summary.scalar("CTC  accuracy",  t_ctc_acc,  step=epoch)
                tf.summary.scalar("Char accuracy",  t_char_acc, step=epoch)
                tf.summary.scalar("Num  error   ",  t_num,      step=epoch)
                tf.summary.scalar("Kor  error   ",  t_kor,      step=epoch)

    def calc_ctc_acc(self, decoded_labels, y_true):
        # Calculate CTC accuracy by comparing predicted labels with true labels
        num_size = len(y_true)
        mum_re_size = num_size
        num_ctc_correct, num_char_correct = 0, 0
        total_ctc, total_char, total_re = 0, 0, 0
        t_error_char, t_error_num, t_error_kor = 0, 0, 0

        for i in range(num_size):
            true_label = [int(label) for label in y_true[i]
                          if label != self.blank_index and label != self.blank_index-1]
            pred_label = [int(label) for label in decoded_labels[i].numpy()
                          if label != -1 and label != self.blank_index-1]

            if pred_label == true_label:
                num_ctc_correct += 1
            else:
                _pred_label = num2char(pred_label, dict_path="data/label.names")
                if is_valid_label(_pred_label):
                    mum_re_size -= 1

            if len(true_label) == len(pred_label):
                total_char += len(true_label)
                for j in range(len(pred_label)):
                    if true_label[j] == pred_label[j]:
                        num_char_correct += 1
                    else:
                        t_error_char += 1
                        if pred_label[j] > 10:
                            t_error_kor += 1
                        else:
                            t_error_num += 1

        total_ctc = num_ctc_correct / num_size
        total_char = num_char_correct / (total_char + 1e-8)
        total_re = num_ctc_correct / (mum_re_size + 1e-8)

        t_num = t_error_num / (t_error_char + 1e-8)
        t_kor = t_error_kor / (t_error_char + 1e-8)

        return total_ctc, total_char, total_re, t_num, t_kor

# main
if __name__ == "__main__":
    from model_fast import *
    # from model import TinyLPR
    from dataloader import DataLoader
    # cpu only
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    seed = 2023
    random.seed = seed
    np.random.seed = seed
    tf.random.set_seed(seed)
    #############################################
    time_steps = 16
    len_label = 8
    n_feat = 64 # 96
    width_multiplier = 0.25 # 1.0
    img_shape = (64, 128, 1)
    num_class = len(load_dict())

    # load checkpoint from latest file
    ckpt_ = glob.glob(os.path.join('checkpoints/', '*.*'))
    ckpt_.sort(key=lambda x: os.path.getmtime(x))
    ckpt_path = ckpt_[-1]
    ckpt_path = "checkpoints/backup/ctc_0.9915_char_0.9989.h5"
    # ckpt_path = "checkpoints/backup/ctc_0.9951_char_0.9993.h5"

    model = TinyLPR(
        time_steps=time_steps,
        n_class=num_class+1,
        n_feat=n_feat,
        width_multiplier=width_multiplier,
        train=True
    ).build(img_shape)
    model.load_weights(ckpt_path, by_name=True, skip_mismatch=True)
    print("load checkpoint from {}".format(ckpt_path))

    val_loader = DataLoader(
        "/Users/haoyu/Downloads/lpr/val",
        img_shape=img_shape,
        time_steps=time_steps,
        label_len=len_label,
        num_class=num_class,
        batch_size=1024,
        shuffle=False,
        data_augmentation=False,
    )

    # test callback
    callback = CTCAccuracyCallback(
        validation_data=val_loader,
        log_dir=os.path.join("logs"),
        blank_index=num_class,
        mode='val')
    callback.set_model(model)
    callback.on_epoch_end(0)
