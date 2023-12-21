from models.primitives import new_conv2d
import tensorflow as tf

def model_13(WIDTH, HEIGHT):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3))
    conv2d = new_conv2d(input, 80, (12, 8), strides=(6, 4))
    conv2d = new_conv2d(conv2d, 128, (7, 7), strides=(4, 4))
    conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(4, 4))
    conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(4, 4))

    flat = tf.keras.layers.Flatten()(conv2d)

    dense = tf.keras.layers.Dropout(0.2)(tf.keras.layers.Dense(128, activation="relu")(flat))
    dense = tf.keras.layers.Dropout(0.2)(tf.keras.layers.Dense(128, activation="relu")(dense))

    output = tf.keras.layers.Dense(5, kernel_initializer="he_uniform", activation="softmax")(dense)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model