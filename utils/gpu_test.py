# test tf gpu
import tensorflow as tf
from tensorflow.python.client import device_lib
# check info of gpu is visible or not on Windows
print(tf.config.list_physical_devices('GPU'))
print(device_lib.list_local_devices())

print(tf.test.is_built_with_cuda())
print(tf.test.is_built_with_gpu_support())

print(tf.sysconfig.get_build_info())

# tf version should under 2.10 on Windows
print(tf.__version__)

# gpu num
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
