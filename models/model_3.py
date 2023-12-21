from models.primitives import new_conv2d
from models.primitives import new_res_block_collection_v2
import tensorflow as tf

def model_3(WIDTH, HEIGHT):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3))
    conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
    maxpool = tf.keras.layers.MaxPooling2D()(conv2d)
    res = new_res_block_collection_v2(3, maxpool, 64)
    res = new_res_block_collection_v2(4, res, 128, first_strides=(2, 2))
    res = new_res_block_collection_v2(6, res, 256, first_strides=2)
    res = new_res_block_collection_v2(3, res, 512, first_strides=2)
    flat = tf.keras.layers.Flatten()(res)
        # dense = new_dense(flat, 4096)
        # dense = new_dense(dense, 2048)
        # dense = new_dense(dense, 1024)
        # dense = new_dense(dense, 512)
        # dense = new_dense(dense, 256)
        # dense = new_dense(dense, 128)
    dense = tf.keras.layers.Dense(5, kernel_initializer="he_uniform", activation="softmax")(flat)
    model = tf.keras.models.Model(inputs=input, outputs=dense)
    return model