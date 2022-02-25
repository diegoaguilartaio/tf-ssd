import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from .header import get_head_from_outputs

def get_model(hyper_params):
    """Generating ssd model for hyper params.
    inputs:
        hyper_params = dictionary

    outputs:
        ssd_model = tf.keras.model
    """
    img_size = hyper_params["img_size"]
    base_model = MobileNetV2(include_top=False, input_shape=(img_size, img_size, 3), alpha=0.5)
    input = base_model.input
    first_conv = base_model.get_layer("block_13_expand_relu").output
    second_conv = base_model.output
    return Model(inputs=input, outputs=[second_conv])

def init_model(model):
    """Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model

    """
    model(tf.random.uniform((1, 300, 300, 3)))
