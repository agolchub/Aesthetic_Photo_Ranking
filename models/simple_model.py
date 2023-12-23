from models.primitives import new_conv2d, new_dense
import tensorflow as tf

def simple_model(WIDTH, HEIGHT):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3))
    do0 = new_conv2d(input, 96, (7, 7), (5, 5))
    do0 = new_conv2d(do0,256,(5,5),(3,3))
    do0 = new_conv2d(do0,384,(3,3),(2,2))

    f1 = tf.keras.layers.Flatten()(do0)

    do1 = new_dense(f1, 4096, dropout_rate=0.2)
    do2 = new_dense(do1, 4096, dropout_rate=0.2)
    do2 = new_dense(do2, 1024, dropout_rate=0.2)
    d4 = tf.keras.layers.Dense(1, kernel_initializer="he_uniform", activation="sigmoid")(do2)
    model = tf.keras.models.Model(inputs=input, outputs=d4)

    return model