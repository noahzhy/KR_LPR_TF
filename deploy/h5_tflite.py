# it's a script to quantization model to specific data type
import os
import sys
import glob
import time
import random
import shutil

import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *
from model.model_fast import *


# use cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# gpu visible
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# tf.keras.backend.set_learning_phase(0)
# tf.compat.v1.enable_eager_execution()


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


def saved_model2pb(
        saved_model_dir,
        input_shape=(1,64,128,1),
        input_node="input0",
        output_node="ctc",
    ):
    # path of the directory where you want to save your model
    frozen_out_path = 'tmp_pb'
    # name of the .pb file
    frozen_graph_filename = "frozen_graph"

    # tf.keras.backend.set_learning_phase(0)

    time_steps = 16
    img_shape = (64, 128, 1)
    batch_size = 1
    #############################################

    # # set model
    model = TinyLPR(
        time_steps=time_steps,
        n_class=69,
        train=False,
    ).build(img_shape)
    model.load_weights(saved_model_dir, by_name=True, skip_mismatch=True)
    # model = tf.keras.models.Model(inputs=model.inputs, outputs=model.outputs[-1])

    # print model inputs and outputs
    print(model.inputs)
    print(model.outputs)

    # model = tf.keras.models.load_model(
    #     saved_model_dir,
    #     custom_objects={
    #         'MobileNetV3Small': MobileNetV3Small,
    #         'LRASPP': LRASPP,
    #     }
    # )
    # model = tf.keras.models.Model(
    #     model.get_layer(name=input_node).input,
    #     model.get_layer(name=output_node).output,
    # )

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(
            model.inputs[0].shape,
            model.inputs[0].dtype
        )
    )

    # check input and output
    print(full_model.outputs[0])

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    print("Frozen model outputs after freezing:", frozen_func.outputs)
    frozen_func.graph.as_graph_def()

    print("Frozen model inputs: ", frozen_func.inputs)
    print("Frozen model outputs:", frozen_func.outputs)

    # Save frozen graph to disk
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=frozen_out_path,
        name=f"{frozen_graph_filename}.pb",
        as_text=False
    )

    input_name  = [input_node.name.split(':')[0]  for input_node  in frozen_func.inputs ]
    output_name = [output_node.name.split(':')[0] for output_node in frozen_func.outputs]

    return frozen_out_path, input_name, output_name


def quantization2tflite(
        model_path,
        input_node="x",
        output_node="Identity",
        mode="pb",
        quantization_mode=tf.uint8,
        save_name="model_uint8",
        representative_dataset=None,
        save_dir="save",
    ):
    assert mode in ["pb", "saved_model", "h5"]
    
    if mode == "h5":
        # load model
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                'MobileNetV3Small': MobileNetV3Small,
                'LRASPP': LRASPP,
            }
        )
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if mode == "saved_model":
        converter = tf.lite.TFLiteConverter.from_saved_model(
            model_path,
            signature_keys=['serving_default']
        )

    if mode == "pb":
        # Convert the model
        converter = tfv1.lite.TFLiteConverter.from_frozen_graph(
            graph_def_file='{}/frozen_graph.pb'.format(model_path),
            input_arrays=input_node,
            output_arrays=output_node,
        )
    
    # converter.experimental_enable_resource_variables = True
    # only for test
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    converter.inference_input_type = quantization_mode  # or tf.int8
    converter.inference_output_type = quantization_mode  # or tf.int8
    tflite_model = converter.convert()
    # get dtype
    dtype = converter.inference_input_type.name
    open('{}/{}_{}.tflite'.format(save_dir, save_name, dtype), 'wb').write(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_type = interpreter.get_input_details()[0]
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]
    print('output: ', output_type)


if __name__ == '__main__':
    # width, height
    IMG_SIZE = (64, 128)
    MODEL_PATH = 'checkpoints/backup/ctc_0.9910_char_0.9988.keras'
    QUANTIZATION_SAMPLE_SIZE = 500
    VAL_DIR = "/home/noah/datasets/val"

    quantization_dataset = RepresentativeDataset(VAL_DIR, IMG_SIZE, QUANTIZATION_SAMPLE_SIZE)

    pb_path, input_name, output_name = saved_model2pb(MODEL_PATH)
    quantization2tflite(
        pb_path, input_name, output_name,
        mode="pb",
        quantization_mode=tf.uint8,
        save_dir='save',
        save_name='tiny_lpr',
        representative_dataset=quantization_dataset,
    )
    # remove pb path folder at last
    shutil.rmtree(pb_path)

    # quantization2tflite(
    #     MODEL_PATH,
    #     mode="h5",
    #     quantization_mode=tf.uint8,
    #     save_dir='save',
    #     save_name='tiny_lpr',
    #     representative_dataset=quantization_dataset,
    # )

    # quantization2tflite(
    #     "save/tinyLPR_deploy",
    #     mode='saved_model',
    #     quantization_mode=tf.uint8,
    #     save_dir='save',
    #     save_name='tiny_lpr',
    #     representative_dataset=quantization_dataset,
    # )
