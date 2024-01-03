import os
import sys
import glob
import random

import jax
import jax.numpy as jnp
from jax.experimental import jax2tf
from PIL import Image
import tensorflow as tf

sys.path.append("./model")
sys.path.append("./utils")
from jax_model import TinyLPR
from utils import *


class RepresentativeDataset:
    def __init__(self, val_dir, img_size=(64, 128), sample_size=300):
        self.val_dir = val_dir
        self.img_size = img_size
        self.sample_size = sample_size
        self.representative_list = random.sample(
            glob.glob(os.path.join(self.val_dir, '*.jpg')),
            self.sample_size
        )

    def __call__(self):
        for image_path in self.representative_list:
            print(image_path)
            input_data = Image.open(image_path).convert('L')
            h, w = self.img_size
            input_data = cv2_imread(image_path)
            input_data = center_fit(cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY), w, h, inter=cv2.INTER_AREA, top_left=True)
            input_data = np.reshape(input_data, (1, *input_data.shape, 1))
            input_data = input_data.astype('float32') / 255.0
            yield [input_data]


model = TinyLPR(train=False)
batch = jnp.ones((1, 64, 128, 1))
v = model.init(jax.random.PRNGKey(0), batch)


def predict(input_img):
    '''
    Function to predict the output from the JAX model
    '''
    return model.apply(v, input_img, mutable=['batch_stats'])


tf_predict = tf.function(
    jax2tf.convert(predict, enable_xla=False),
    input_signature=[
        tf.TensorSpec(
            shape=[1, 64, 128, 1],
            dtype=tf.float32,
            name='input_image'
        )
    ],
    autograph=False)

IMG_SIZE = (64, 128)
QUANTIZATION_SAMPLE_SIZE = 10
VAL_DIR = "data/val"
quantization_dataset = RepresentativeDataset(VAL_DIR, IMG_SIZE, QUANTIZATION_SAMPLE_SIZE)
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tf_predict.get_concrete_function()], tf_predict)

converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.representative_dataset = quantization_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

with open('./quantized.tflite', 'wb') as f:
    f.write(tflite_quantized_model)
