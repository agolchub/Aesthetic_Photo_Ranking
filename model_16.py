from models.primitives import new_conv2d
from models.primitives import new_res_block_collection_v2
import tensorflow as tf

def model_16(WIDTH, HEIGHT):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3))
    conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
    maxpool = tf.keras.layers.MaxPooling2D()(conv2d)
    res1 = new_res_block_collection_v2(3, maxpool, 64)
    res2 = new_res_block_collection_v2(4, res1, 128, first_strides=(2, 2))
    res3 = new_res_block_collection_v2(6, res2, 256, first_strides=2)
    res4 = new_res_block_collection_v2(3, res3, 512, first_strides=2)

    res1 = new_conv2d(res1, 128, (3, 3), strides=(2, 2), padding="same")
    res1 = new_conv2d(res1, 256, (2, 2), strides=(2, 2), padding="same")
    res1 = new_conv2d(res1, 512, (2, 2), strides=(2, 2), padding="same")
    res1 = new_conv2d(res1, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    res2 = new_conv2d(res2, 256, (3, 3), strides=(2, 2), padding="same")
    res2 = new_conv2d(res2, 512, (2, 2), strides=(2, 2), padding="same")
    res2 = new_conv2d(res2, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    res3 = new_conv2d(res3, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)
    res3 = new_conv2d(res3, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    res4 = new_conv2d(res4, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    add = tf.keras.layers.Add()([res1, res2, res3, res4])
    res_combined = tf.keras.layers.Activation("relu")(add)

    conv2d = new_conv2d(res_combined, 512, (3, 3), strides=(3, 2), dropout_rate=0.2)
    conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), dropout_rate=0.2)

    flat = tf.keras.layers.Flatten()(conv2d)

    output = tf.keras.layers.Dense(5, kernel_initializer="he_uniform", activation="linear")(flat)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model