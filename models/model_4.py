from models.primitives import new_conv2d, new_dense
from models.primitives import new_res_block_collection_v2
import tensorflow as tf

def model_4(WIDTH, HEIGHT):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3))
    conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
    maxpool = tf.keras.layers.MaxPooling2D()(conv2d)
    res1 = new_res_block_collection_v2(3, maxpool, 64)
    res2 = new_res_block_collection_v2(4, res1, 128, first_strides=(2, 2))
    res3 = new_res_block_collection_v2(6, res2, 256, first_strides=2)
    res4 = new_res_block_collection_v2(3, res3, 512, first_strides=2)

    res1 = new_conv2d(res1, 128, (3, 3), strides=(2, 2))
    res1 = new_conv2d(res1, 256, (3, 3), strides=(2, 2))
    res1 = new_conv2d(res1, 512, (3, 3), strides=(2, 2))

    res2 = new_conv2d(res2, 256, (3, 3), strides=(2, 2))
    res2 = new_conv2d(res2, 512, (3, 3), strides=(2, 2))

    res3 = new_conv2d(res3, 512, (3, 3), strides=(2, 2))

    flat1 = tf.keras.layers.Flatten()(res1)
    flat2 = tf.keras.layers.Flatten()(res2)
    flat3 = tf.keras.layers.Flatten()(res3)
    flat4 = tf.keras.layers.Flatten()(res4)

    dense1 = new_dense(flat1, 16)
    dense2 = new_dense(flat2, 16)
    dense3 = new_dense(flat3, 16)
    dense4 = new_dense(flat4, 16)

    concat = tf.keras.layers.Concatenate()([dense1, dense2, dense3, dense4])

    output = tf.keras.layers.Dense(1, kernel_initializer="he_uniform", activation="linear")(concat)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model