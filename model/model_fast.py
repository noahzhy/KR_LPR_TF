import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Sequential
from keras_flops import get_flops


class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__()

        self.concat = Concatenate(axis=-1)
        self.conv = Conv2D(1, kernel_size=(1, 1), strides=1, padding="same",
                kernel_initializer='he_normal',
                use_bias=False,
                activation='sigmoid')

    def call(self, inputs, **kwargs):
        _avg = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        _max = tf.reduce_max(inputs, axis=-1, keepdims=True)
        cat = self.concat([_max, _avg])
        x = self.conv(cat)
        mul = tf.multiply(x, inputs)
        return mul


class Attention(Layer):
    def __init__(self, C=128, T=16, batch_size=-1, **kwargs):
        super(Attention, self).__init__()

        self.T = T
        self.C = C
        self.bs = batch_size

        self.spatial_attention = SpatialAttention()
        self.channel_attention = Sequential([
            GlobalAveragePooling2D(keepdims=True),
            Conv2D(filters=C, kernel_size=(1, 1), strides=1, padding="same",
                kernel_initializer='he_normal',
                use_bias=False,
                activation='sigmoid'),
        ], name="channel_attention")

        # 1x1 conv
        self.conv = Conv2D(filters=T, kernel_size=(1, 1), strides=1, padding="same",
                kernel_initializer='he_normal',
                activation='relu',
                name="conv_char")

    def call(self, inputs, **kwargs):
        _channel = self.channel_attention(inputs)
        mul = tf.multiply(inputs, _channel)
        _spatial = self.spatial_attention(mul)

        char_map = self.conv(inputs)

        x = tf.reshape(char_map, (self.bs, 128, self.T))        # [bs, H*W, T]
        y = tf.reshape(_spatial, (self.bs, 128, self.C))        # [bs, H*W, C]
        out = tf.einsum('ijk,ijl->ikl', x, y, name='mat')       # [bs, T, C]

        return out, char_map

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            'C': self.C,
            'T': self.T,
            'batch_size': self.bs,
        })
        return config


def upsample(x, up_size=2):
    nn = Sequential()
    for _ in range(np.log2(up_size).astype(np.int32)):
        block = Sequential([
            UpSampling2D(size=(2, 2), interpolation='bilinear'),
            Conv2D(64, 5, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
            PReLU(shared_axes=[1, 2]),
        ])
        nn.add(block)

    return nn(x)


class BottleNeck(Layer):
    def __init__(self, in_size, exp_size, out_size, s, k=3, activation=tf.nn.relu6, scale=1.0, name="block", **kwargs):
        super(BottleNeck, self).__init__()
        self.stride = s
        self.in_size = _make_divisible(in_size * scale, 8)
        self.exp_size = _make_divisible(exp_size * scale, 8)
        self.out_size = _make_divisible(out_size * scale, 8)

        self.pw_in = Sequential([
            Conv2D(filters=self.exp_size, kernel_size=(1, 1), strides=1, padding="same", kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(momentum=0.99),
            Activation(activation)
        ], name="{}.pw_in".format(name))

        self.dw = Sequential([
            DepthwiseConv2D(kernel_size=(k, k), strides=s, padding="same", use_bias=False),
            BatchNormalization(momentum=0.99),
            Activation(activation),
        ], name="{}.dw".format(name))

        self.pw_out = Sequential([
            Conv2D(filters=self.out_size, kernel_size=(1, 1), strides=1, padding="same", kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(momentum=0.99),
        ], name="{}.pw_out".format(name))

    def call(self, inputs, **kwargs):
        # shortcut
        x = self.pw_in(inputs)
        x = self.dw(x)
        x = self.pw_out(x)

        if self.stride == 1 and self.in_size == self.out_size:
            x = add([x, inputs])

        return x


def _make_divisible(v, divisor, min_value=16):
    """https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class MobileNetV3Small(Layer):
    def __init__(self, width_multiplier=1.0, out_channels=128, **kwargs):
        super(MobileNetV3Small, self).__init__()
        self.width_multiplier = width_multiplier
        self.out_channels = out_channels

        self.conv = Sequential([
            Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, kernel_initializer='he_normal'),
            BatchNormalization(trainable=True, momentum=0.99),
            Activation(tf.nn.relu),
        ], name="conv1")

        # Bottleneck layers
        bnecks = [
            # k  in    exp     out      NL          s
            [3,  16,    16,     16,     "relu",     2],
            [3,  16,    72,     24,     "relu",     1],
            [3,  24,    88,     24,     "relu",     1],
            [5,  24,    96,     40,     "relu6",    2],
            [5,  40,    240,    40,     "relu6",    1],
            [5,  40,    120,    48,     "relu6",    1],
            [5,  48,    144,    48,     "relu6",    1],
        ]

        self.bneck = Sequential(name="bneck")
        for i, (k, _in, exp, out, NL, s) in enumerate(bnecks):
            self.bneck.add(BottleNeck(_in, exp, out, s, k, NL, width_multiplier, name="bneck.{}".format(i)))

        exp = _make_divisible(288 * width_multiplier, 8)
        # exp = 72
        self.last = BottleNeck(16, exp, out_channels, 1, 5, tf.nn.relu, name="last")

    def call(self, inputs, training=True, **kwargs):
        x = self.conv(inputs, training=training)
        x = self.bneck(x, training=training)
        x = self.last(x, training=training)
        return x

    def get_config(self):
        config = super(MobileNetV3Small, self).get_config()
        config.update({
            'width_multiplier': self.width_multiplier,
            'out_channels': self.out_channels,
        })
        return config


class TinyLPR(Model):
    def __init__(self,
        time_steps=16,
        n_class=69,
        n_feat=96,
        width_multiplier=1.0,
        train=True,
        **kwargs):
        super(TinyLPR, self).__init__()
        self.n_class = n_class
        self.n_feat = n_feat
        self.train = train
        self.time_steps = time_steps
        # backbone model
        self.backbone = MobileNetV3Small(width_multiplier, n_feat)
        # head
        self.attn = Attention(n_feat, time_steps, batch_size=-1 if train else 1)
        # ctc
        self.dense = Dense(self.n_class, kernel_initializer='he_normal', name='dense')
        self.softmax = Activation(tf.nn.softmax, name='ctc')

    def build(self, input_shape):
        inputs = Input(shape=input_shape, name='input0')

        f_map = self.backbone(inputs, training=self.train)
        mat, c_map = self.attn(f_map)
        dense = self.dense(mat)
        ctc = self.softmax(dense)

        if self.train:
            # 8x upsample
            mask = upsample(c_map, up_size=8)
            mask = Conv2D(self.time_steps, 1, strides=1, padding='same', kernel_initializer='he_normal', name='conv_1x1')(mask)
            mask = Activation(tf.nn.sigmoid, name='mask')(mask)

            # # concat mat and ctc
            mat_ctc = Concatenate(axis=-1, name='mat_ctc')([mat, ctc])
            # mat_ctc = dense

            return Model(
                inputs=[inputs],
                outputs=[mask, mat_ctc, ctc],
                name='tinyLPR',
            )

        return Model(
            inputs=[inputs],
            outputs=[ctc],
            name='tinyLPR',
        )

    def get_config(self):
        config = super(TinyLPR, self).get_config()
        config.update({
            'time_steps': self.time_steps,
            'n_class': self.n_class,
            'n_feat': self.n_feat,
            'train': self.train,
        })
        return config


if __name__ == '__main__':
    # disable gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    input_shape = (64, 128, 1)
    mask_len = 8
    n_feat = 96
    width_multiplier = 1.0

    # train model
    model = TinyLPR(
        width_multiplier=width_multiplier,
        n_feat=n_feat,
        train=True,
    ).build(input_shape)
    model.summary()

    # deploy model
    model = TinyLPR(
        width_multiplier=width_multiplier,
        n_feat=n_feat,
        train=False,
    ).build(input_shape)
    model.summary()

    # get flops
    flops = get_flops(model, 1)
    flops = round(flops / 10.0 ** 6, 2)
    print(f"FLOPS: {flops} M")

    # # save as tflite
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # tflite_model = converter.convert()
    # with open("save/tinyLPR.tflite", 'wb') as f:
    #     f.write(tflite_model)
