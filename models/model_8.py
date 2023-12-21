from models.primitives import new_conv2d
import tensorflow as tf

def model_8(WIDTH, HEIGHT):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3))
    conv2d = new_conv2d(input, 64, (35, 35), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 128, (17, 17), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 256, (9, 9), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")
    conv2d_1 = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")

    conv2d = new_conv2d(input, 64, (49, 49), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 128, (23, 23), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 256, (11, 11), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 512, (5, 5), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")
    conv2d_2 = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")

    conv2d = new_conv2d(input, 64, (7, 7), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 128, (5, 5), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")
    conv2d_3 = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")

    conv2d = new_conv2d(input, 64, (128, 128), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 128, (23, 23), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 256, (7, 7), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 512, (5, 5), strides=(2, 2), padding="same")
    conv2d = new_conv2d(conv2d, 512, (5, 5), strides=(2, 2), padding="same")
    conv2d_4 = new_conv2d(conv2d, 512, (5, 5), strides=(2, 2), padding="same")

    flat_1 = tf.keras.layers.Flatten()(conv2d_1)
    flat_2 = tf.keras.layers.Flatten()(conv2d_2)
    flat_3 = tf.keras.layers.Flatten()(conv2d_3)
    flat_4 = tf.keras.layers.Flatten()(conv2d_4)

    concat = tf.keras.layers.Concatenate()([flat_1, flat_2, flat_3, flat_4])

    output = tf.keras.layers.Dense(1, kernel_initializer="he_uniform", activation="linear")(concat)
    model = tf.keras.models.Model(inputs=input, outputs=output)
    return model