import tensorflow as tf
from tensorflow.keras import initializers, models, optimizers, layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Concatenate, Flatten

import sys


def init_layer(layer):
    try:
        initializer = tf.keras.initializers.GlorotUniform()
        import keras.backend as K
        session = K.get_session()
        if hasattr(layer, 'kernel_initializer'):
            print("initializing kernel weights")
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            print("initializing bias weights")
            layer.bias.initializer.run(session=session)
        print(layer.name, " re-initilized")
    except:
        print(layer.name, " could not be re-initilized", sys.exc_info())


def new_conv2d(input, n, size=(2, 2), strides=(2, 2), activation="relu", kernel_initializer="glorot_uniform",
               batch_normalization=True, dropout_rate=0.2, padding="valid"):
    conv2d = layers.Conv2D(n, size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, dtype=tf.float16)(input)
    batch_normalization_layer = layers.BatchNormalization(dtype=tf.float16)(conv2d) if batch_normalization else conv2d
    activation_layer = layers.Activation(activation, dtype=tf.float16)(batch_normalization_layer) if activation is not None else batch_normalization_layer
    dropout = layers.Dropout(dropout_rate, dtype=tf.float16)(activation_layer) if dropout_rate > 0 else activation_layer

    return dropout


def new_dense(input, n, activation="relu", kernel_initializer="glorot_uniform", dropout_rate=0.2):
    dense = layers.Dense(n, activation=activation, kernel_initializer=kernel_initializer, dtype=tf.float16)(input)
    dropout = layers.Dropout(dropout_rate, dtype=tf.float16)(dense)
    return dropout


def new_res_block(input, n, m, size, strides):
    conv2d = layers.Conv2D(n, size, strides=strides, dtype=tf.float16)(input)
    batchNormalization = layers.BatchNormalization(dtype=tf.float16)(conv2d)
    activation = layers.Activation("relu", dtype=tf.float16)(batchNormalization)
    shortcut = activation
    conv2d = layers.Conv2D(n * m, (2, 2), strides=(1, 1), dtype=tf.float16)(activation)
    batchNormalization = layers.BatchNormalization(dtype=tf.float16)(conv2d)
    activation = layers.Activation("relu", dtype=tf.float16)(batchNormalization)
    conv2d = layers.Conv2D(n * m * m, (2, 2), strides=(1, 1), dtype=tf.float16)(activation)
    batchNormalization = layers.BatchNormalization(dtype=tf.float16)(conv2d)

    shortcut = layers.Conv2D(n * m * m, (3, 3), strides=(1, 1), dtype=tf.float16)(shortcut)
    shortcut = layers.BatchNormalization(dtype=tf.float16)(shortcut)

    add = layers.Add(dtype=tf.float16)([batchNormalization, shortcut])
    activation = layers.Activation("relu", dtype=tf.float16)(add)
    return activation


def new_res_block_v2(input, n, size=3, strides=1, first_strides=1):
    shortcut = input
    if (first_strides != strides):
        shortcut = layers.Conv2D(n, (1, 1), strides=first_strides, padding='same', activation=None, dtype=tf.float16)(shortcut)
    conv2d = layers.Conv2D(n, size, strides=first_strides, padding='same', activation=None, dtype=tf.float16)(input)
    batch_normalization = layers.BatchNormalization(dtype=tf.float16)(conv2d)
    activation = layers.Activation("relu", dtype=tf.float16)(batch_normalization)
    conv2d = layers.Conv2D(n, size, strides=strides, padding='same', activation=None, dtype=tf.float16)(activation)
    batch_normalization = layers.BatchNormalization(dtype=tf.float16)(conv2d)
    add = layers.Add(dtype=tf.float16)([batch_normalization, shortcut])
    activation = layers.Activation("relu", dtype=tf.float16)(add)
    return activation

def new_res_block_collection_v2(blocks, input, n, size=3, strides=1, first_strides=1):
    activation = input
    for i in range(blocks):
        print(i)
        activation = new_res_block_v2(activation, n, size, strides, strides if i != 0 else first_strides)
    return activation