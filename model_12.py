from models.primitives import new_conv2d
import tensorflow as tf

def model_12(WIDTH, HEIGHT):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3))
    conv2d = new_conv2d(input, 80, (102, 68), strides=(3, 2))
    conv2d = new_conv2d(conv2d, 128, (7, 7), strides=(2, 2))
    conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(1, 1))
    conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2))
    conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(1, 1))
    conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(2, 2))
    conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(1, 1))
    conv2d = new_conv2d(conv2d, 128, (3, 3), strides=(2, 2))
    conv2d = new_conv2d(conv2d, 128, (3, 3), strides=(1, 1))

    flat = tf.keras.layers.Flatten()(conv2d)

    dense = tf.keras.layers.Dense(128, activation="sigmoid")(flat)
    dense = tf.keras.layers.Dense(128, activation="sigmoid")(dense)

    output = tf.keras.layers.Dense(5, kernel_initializer="he_uniform", activation="softmax")(dense)
    model = tf.keras.models.Model(inputs=input, outputs=output)