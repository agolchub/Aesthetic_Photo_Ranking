from models.primitives import new_conv2d
from models.primitives import new_res_block_collection_v2
import tensorflow as tf


def model_20(WIDTH, HEIGHT):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3),dtype=tf.float16)
    conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
    maxpool = tf.keras.layers.MaxPooling2D(dtype=tf.float16)(conv2d)
    res1 = new_res_block_collection_v2(3, maxpool, 64)
    res2 = new_res_block_collection_v2(4, res1, 128, first_strides=(2, 2))
    res3 = new_res_block_collection_v2(6, res2, 256, first_strides=2)
    res4 = new_res_block_collection_v2(3, res3, 512, first_strides=2)
    res5 = new_res_block_collection_v2(3, res4, 512, first_strides=2)
    res6 = new_res_block_collection_v2(3, res5, 512, first_strides=2)
    res7 = new_res_block_collection_v2(3, res6, 512, first_strides=2)
    res8 = new_res_block_collection_v2(3, res7, 512, first_strides=2)

    res1 = new_conv2d(res1, 128, (3, 3), strides=(2, 2), padding="same")
    res1 = new_conv2d(res1, 256, (2, 2), strides=(2, 2), padding="same")
    res1 = new_conv2d(res1, 512, (2, 2), strides=(2, 2), padding="same")
    res1 = new_conv2d(res1, 512, (2, 2), strides=(2, 2), padding="same")
    res1 = new_conv2d(res1, 512, (2, 2), strides=(2, 2), padding="same")
    res1 = new_conv2d(res1, 512, (2, 2), strides=(2, 2), padding="same")
    res1 = new_conv2d(res1, 1024, (2, 2), strides=(2, 2), padding="same")
    res1 = new_conv2d(res1, 1024, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    res2 = new_conv2d(res2, 256, (3, 3), strides=(2, 2), padding="same")
    res2 = new_conv2d(res2, 512, (2, 2), strides=(2, 2), padding="same")
    res2 = new_conv2d(res2, 512, (2, 2), strides=(2, 2), padding="same")
    res2 = new_conv2d(res2, 512, (2, 2), strides=(2, 2), padding="same")
    res2 = new_conv2d(res2, 512, (2, 2), strides=(2, 2), padding="same")
    res2 = new_conv2d(res2, 1024, (2, 2), strides=(2, 2), padding="same")
    res2 = new_conv2d(res2, 1024, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    res3 = new_conv2d(res3, 512, (2, 2), strides=(2, 2), padding="same")
    res3 = new_conv2d(res3, 512, (2, 2), strides=(2, 2), padding="same")
    res3 = new_conv2d(res3, 512, (2, 2), strides=(2, 2), padding="same")
    res3 = new_conv2d(res3, 512, (2, 2), strides=(2, 2), padding="same")
    res3 = new_conv2d(res3, 1024, (2, 2), strides=(2, 2), padding="same")
    res3 = new_conv2d(res3, 1024, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    res4 = new_conv2d(res4, 512, (2, 2), strides=(2, 2), padding="same")
    res4 = new_conv2d(res4, 512, (2, 2), strides=(2, 2), padding="same")
    res4 = new_conv2d(res4, 512, (2, 2), strides=(2, 2), padding="same")
    res4 = new_conv2d(res4, 1024, (2, 2), strides=(2, 2), padding="same")
    res4 = new_conv2d(res4, 1024, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    res5 = new_conv2d(res5, 512, (2, 2), strides=(2, 2), padding="same")
    res5 = new_conv2d(res5, 512, (2, 2), strides=(2, 2), padding="same")
    res5 = new_conv2d(res5, 1024, (2, 2), strides=(2, 2), padding="same")
    res5 = new_conv2d(res5, 1024, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    res6 = new_conv2d(res6, 512, (2, 2), strides=(2, 2), padding="same")
    res6 = new_conv2d(res6, 1024, (2, 2), strides=(2, 2), padding="same")
    res6 = new_conv2d(res6, 1024, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    res7 = new_conv2d(res7, 512, (2, 2), strides=(2, 2), padding="same")
    res7 = new_conv2d(res7, 1024, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    res8 = new_conv2d(res8, 1024, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

    print(f'shape: {res1.shape}')
    print(f'shape: {res2.shape}')
    print(f'shape: {res3.shape}')
    print(f'shape: {res4.shape}')
    print(f'shape: {res5.shape}')
    print(f'shape: {res6.shape}')
    print(f'shape: {res7.shape}')
    print(f'shape: {res8.shape}')

    add = tf.keras.layers.Add(dtype=tf.float16)([res1, res2, res3, res4, res5, res6, res7, res8])
    res_combined = tf.keras.layers.Activation("relu", dtype=tf.float16)(add)

    conv2d = new_conv2d(res_combined, 2048, (2, 2), strides=(1, 1), dropout_rate=0.2)
    conv2d = new_conv2d(conv2d, 2048, (1, 1), strides=(1, 1), dropout_rate=0.2)

    flat = tf.keras.layers.Flatten(dtype=tf.float16)(conv2d)

    output = tf.keras.layers.Dense(10, kernel_initializer="he_uniform", activation="softmax", dtype=tf.float16)(flat)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model