import os, time, math
from itertools import groupby

import tensorflow as tf
from keras.layers import *
from keras import backend as K


eps = K.epsilon()

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = tf.cast(tf.squeeze(label_length, axis=-1), tf.int32)
    input_length = tf.cast(tf.squeeze(input_length, axis=-1), tf.int32)
    sparse_label = tf.cast(K.ctc_label_dense_to_sparse(y_true, label_length), tf.int32)
    # sparse_label = tf.sparse.from_dense(y_true)
    # print('sparse_label', sparse_label)
    # return tf.compat.v1.nn.ctc_loss(
    #     inputs=tf.math.log(y_pred + eps),
    #     labels=sparse_label,
    #     sequence_length=input_length,
    #     time_major=False,
    # )
    return tf.compat.v1.nn.ctc_loss_v2(
        labels=sparse_label,
        logits=tf.math.log(y_pred + eps),
        label_length=None,
        logit_length=input_length,
        logits_time_major=False,
        blank_index=-1,
    )


# Sparse Categorical Crossentropy Loss
class SCELoss(Layer):
    def __init__(self, name="sce_loss", **kwargs):
        super(SCELoss, self).__init__(name=name, **kwargs)
        # ce loss
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            # before softmax
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
        )

    def call(self, y_true, y_pred):
        '''
        y_pred: [bs, T, feat_dims]
        y_true: [bs, T]
        '''
        bs, T, feat_dims = y_pred.get_shape().as_list()

        y_pred = tf.reshape(y_pred, shape=(-1, feat_dims))
        y_true = tf.reshape(y_true, shape=(-1,))
        y_true = tf.cast(y_true, tf.int32)

        loss = self.loss_fn(y_true, y_pred)
        loss = tf.reduce_mean(loss)
        return loss


class FocalCTCLoss(Layer):
    def __init__(self, alpha=2.0, gamma=3.0, name="focal_ctc_loss", **kwargs):
        super(FocalCTCLoss, self).__init__(name=name, **kwargs)
        self.loss_fn = ctc_batch_cost
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        input_length = K.tile([[K.shape(y_pred)[1]]], [K.shape(y_pred)[0], 1])
        label_length = K.tile([[K.shape(y_true)[1]]], [K.shape(y_true)[0], 1])
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        p = tf.exp(-loss)
        focal_ctc_loss = tf.multiply(tf.multiply(self.alpha, tf.pow((1 - p), self.gamma)), loss)
        return tf.reduce_mean(focal_ctc_loss)


class CELoss(Layer):
    def __init__(self, name="ce_loss"):
        super(CELoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE,
        )(y_true, y_pred)
        return loss


class BCELoss(Layer):
    def __init__(self, name="bce_loss"):
        super(BCELoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE,
        )(y_true, y_pred)
        return loss


class SmoothL1Loss:
    def __init__(self, name="smooth_l1_loss"):
        super(SmoothL1Loss, self).__init__(name=name)

    """ Compute smooth l1 loss between the predicted bounding boxes and the ground truth bounding boxes.

    Args:
        - y_true: The ground truth bounding boxes.
        - y_pred: The predicted bounding boxes.
    """

    def call(self, y_true, y_pred, **kwargs):
        absolute_loss = tf.abs(y_true - y_pred)
        square_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = tf.where(tf.less(absolute_loss, 1.0),
                           square_loss, absolute_loss - 0.5)
        return tf.reduce_sum(l1_loss, axis=-1)


class DiceLoss(Layer):
    def __init__(self, name="dice_loss"):
        super(DiceLoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(
            y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        dice = tf.reduce_mean((2.0 * intersection + 1e-7) / (union + 1e-7))
        return 1 - dice


# IOU loss
class IOULoss(Layer):
    def __init__(self, name="iou_loss"):
        super(IOULoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union = tf.reduce_sum(
            y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        iou = tf.reduce_mean((intersection + 1e-7) /
                             (union - intersection + 1e-7))
        return 1 - iou


# Dice + BCE
class DiceBCELoss(Layer):
    def __init__(self, name="dice_bce_loss"):
        super(DiceBCELoss, self).__init__(name=name)

    def call(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # flatten label and prediction tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])

        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        dice_loss = 1 - (2. * intersection + 1e-7) / \
            (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7)

        bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE,
        )(y_true, y_pred)

        return dice_loss + bce


class CTCCenterLoss(Layer):
    def __init__(self,
                 n_class=86,
                 feat_dims=96,
                 random_init=False,
                 name="ctc_center_loss",
                 **kwargs):
        super(CTCCenterLoss, self).__init__(name=name, **kwargs)

        self.n_class = n_class
        self.feat_dims = feat_dims
        self.centers = tf.Variable(
            initial_value=tf.zeros(
                shape=(n_class, feat_dims), dtype=tf.float32, name="centers"),
            trainable=False,
            name="centers",
        )
        # random init centers
        if random_init:
            self.centers.assign(tf.random.normal(
                shape=(n_class, feat_dims), mean=0.0, stddev=1.0))

    def call(self, y_true, y_pred, **kwargs):
        # y_pred was concat by features and preds
        # split y_pred
        features = y_pred[:, :, :self.feat_dims]
        preds = y_pred[:, :, self.feat_dims:]

        feats_reshape = tf.reshape(features, shape=(-1, self.feat_dims))
        label = tf.reshape(tf.argmax(preds, axis=-1), shape=(-1,))

        bs = tf.shape(feats_reshape)[0]

        feat = tf.reduce_sum(tf.pow(feats_reshape, 2), axis=1, keepdims=True)
        feat = tf.broadcast_to(feat, shape=(bs, self.n_class))

        center = tf.reduce_sum(tf.pow(self.centers, 2), axis=1, keepdims=True)
        center = tf.broadcast_to(center, shape=(self.n_class, bs))
        center = tf.cast(center, dtype=tf.float32)
        center = tf.transpose(center)

        distmat = tf.add(feat, center)

        feat_dot_center = tf.matmul(feats_reshape, tf.transpose(self.centers))
        distmat = distmat - 2.0 * feat_dot_center

        # mask
        classes = tf.range(self.n_class, dtype=tf.int32)
        label = tf.broadcast_to(tf.expand_dims(
            label, axis=1), shape=(bs, self.n_class))
        mask = tf.math.equal(
            tf.broadcast_to(classes, shape=(bs, self.n_class)),
            tf.cast(label, dtype=tf.int32),
        )
        mask = tf.cast(mask, dtype=tf.float32)

        # compute loss
        dist = tf.multiply(distmat, mask)
        # clamp dist
        dist = tf.clip_by_value(dist, clip_value_min=1e-12, clip_value_max=1e+12)
        loss = tf.reduce_mean(dist)

        return loss


# CenterCTCLoss
class CenterCTCLoss(Layer):
    def __init__(self, alpha=0.05, n_class=85, feat_dims=64, name="center_ctc_loss", **kwargs):
        super(CenterCTCLoss, self).__init__(name=name, **kwargs)
        self.alpha = alpha
        self.n_class = n_class
        # blank label
        self.blank_index = n_class - 1
        self.feat_dims = feat_dims
        self.centers = tf.Variable(
            initial_value=tf.zeros(
                shape=(n_class, feat_dims), dtype=tf.float32, name="centers"),
            trainable=False,
            name="centers",
        )

        # self.char_num = tf.placeholder(tf.int32, shape=[None], name="char_num")
        self.char_num = tf.Variable(
            initial_value=tf.zeros(
                shape=(1,), dtype=tf.int32, name="char_num"),
            trainable=False,
            name="char_num",
        )
        # self.char_pos_init = tf.placeholder(tf.int32, shape=[None, 2], name='char_pos')
        self.char_pos_init = tf.Variable(
            initial_value=tf.zeros(
                shape=(1, 2), dtype=tf.int32, name="char_pos"),
            trainable=False,
            name="char_pos",
        )

    # @tf.function()
    def call(self, y_true, y_pred, **kwargs):
        labels = y_true
        features, preds = y_pred[0], y_pred[1]

        pred_labels = tf.argmax(features, axis=-1, output_type=tf.int32)
        print('pred_labels', pred_labels)

        self.features = features
        self.pred_labels = pred_labels
        self.labels = labels

        self.preds2features(
            self.pred_labels,
            self.labels,
            self.features,
            self.char_num,
            self.char_pos_init
        )

        centers_batch = tf.gather(self.centers, labels)
        # calculate loss
        loss = tf.nn.l2_loss(features - centers_batch)
        # calculate diff
        diff = centers_batch - features
        # unique labels and counts
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        # calculate update centers
        diff = self.alpha * diff / tf.cast((1 + appear_times), tf.float32)
        # update centers
        self.centers.assign_sub(
            tf.scatter_nd(
                tf.expand_dims(labels, axis=1),
                diff,
                shape=(self.n_class, self.feat_dims),
            )
        )
        return loss

    def preds2features(self, preds, labels, features, char_num, poses):
        '''
        preds: output of softmax, shape = [bs, T, n_class]
        labels: shape = [bs, T]
        features: shape = [bs, T, feat_dims]
        '''
        is_char = tf.not_equal(preds, self.blank_index)
        print(is_char)
        # get repeat char
        char_rep = tf.equal(preds[:, :-1], preds[:, 1:])
        print(char_rep)
        tail = tf.greater(preds[:, :1], self.blank_index)
        char_rep = tf.concat([char_rep, tail], axis=1)
        # get last position of repeat char
        char_no_rep = tf.math.logical_and(
            is_char,
            tf.math.logical_not(char_rep))
        self.char_pos, self.char_label = self.get_char_pos_label(
            char_no_rep,
            labels,
            char_num,
            poses
        )
        self.features = self.get_features(self.char_pos, features)

    # @tf.function
    def get_char_pos_label(self, preds, label, char_num, poses):
        """
        过滤掉预测漏字的样本，返回过滤后的字符位置和标签
        Args:
            preds: 去掉重复字符后的预测结果，是字符的位置为 True，否则为 False
            label: 字符标签
            char_num: 每个样本的字符数
            poses: 初始化的字符位置
        Returns:
            字符位置: 2D tensor of shape (num of chars, 2)，后一个维度为（字符位置，图片序号）
            标签：1D tensor of shape (num of chars,)
        """
        i = tf.constant(0, dtype=tf.int32)
        char_total = tf.constant(0, dtype=tf.int32)

        for char in preds:
            char_pos = tf.cast(tf.where(char), tf.int32)
            print('char_pos', char_pos)

            # 判断预测出的字符数和 gt 是否一致，如果不一致则忽略此样本
            char_seg_num = tf.shape(char_pos)[0]
            if not tf.equal(char_seg_num, char_num[i]):
                tf.print('切出的字符数量与真实值不同，忽略此样本：',
                            label[char_total:char_total + char_num[i]], char_seg_num, 'vs', char_num[i], summarize=-1)
                label = tf.concat([label[:char_total], label[char_total + char_num[i]:]], axis=0)
                i = tf.add(i, 1)
                continue
            else:
                char_total = tf.add(char_total, char_num[i])

            # 在seg中添加 batch 序号标识，方便后续获取 feature
            batch_i = char_pos[:, :1]
            batch_i = tf.broadcast_to(i, tf.shape(batch_i))
            char_pos = tf.concat([char_pos, batch_i],
                                 axis=1, name='add_batch_index')

            # concat to one tensor
            poses = tf.concat([poses, char_pos], axis=0, name='push_in_segs')
            i = tf.add(i, 1)

        return poses[1:], label

    @staticmethod
    def get_features(char_pos, features):
        """
        get features from position of temporal step
        Args:
            char_pos: position of char, 2D tensor of shape (num of chars, 2), last dim is position of char
            features: inputs of last fully connected layer
        Returns:
            feat_char: features of chars
        """
        def get_slice(pos):
            feature_one_char = features[pos[1], pos[0], :]
            return feature_one_char

        return tf.map_fn(get_slice, char_pos, dtype=tf.float32)


# main
if __name__ == "__main__":
    # cpu mode
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    loss = FocalCTCLoss()
    y_pred = tf.linspace(0.0, 1.0, 3 * 9 * 69)
    y_pred = tf.reshape(y_pred, shape=(3, 9, 69))
    y_true = tf.constant([
        [67,  2,  5, 31,  2,  8,  3,  7,],
        [58,  3,  7, 45,  6,  7,  5,  2,],
        [67,  6,  0, 17,  5,  0,  0,  5,]], dtype=tf.int32)
    s = time.process_time()
    v = loss(y_true, y_pred)
    print("time: {:.4f} ms".format((time.process_time() - s) * 1000))
    print(v)

    # loss = CTCCenterLoss(feat_dims=96)
    # y_pred = tf.random.uniform(shape=(3, 16, 96))
    # # y_true = tf.random.uniform(shape=(3, 9), maxval=85, dtype=tf.int32)
    # y_ctc = tf.random.uniform(shape=(3, 16, 85))
    # cat = tf.concat([y_pred, y_ctc], axis=-1)
    # v = loss(y_true, cat)
    # print(v)

    # loss = SCELoss()
    # y_pred = tf.random.uniform(shape=(8, 8, 16, 16))
    # y_true = tf.random.uniform(shape=(8, 16,))
    # v = loss(y_true, y_pred)
    # print(v)
