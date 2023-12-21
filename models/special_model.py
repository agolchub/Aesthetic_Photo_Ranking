import tensorflow as tf
from models.primitives import new_conv2d, new_dense


def build_layers_for_model(model, input, unlock_segment_weights):
    activation = "relu"
    dropout_rate = 0.2
    conv2d = tf.keras.layers.Conv2D(80, (20, 20), strides=(5, 5), kernel_initializer="glorot_uniform",
                           weights=model.layers[1].get_weights(), dtype=tf.float16)(input)
    batchNormalization = tf.keras.layers.BatchNormalization(weights=model.layers[2].get_weights(), dtype=tf.float16)(conv2d)
    activationLayer = tf.keras.layers.Activation(activation, weights=model.layers[3].get_weights(), dtype=tf.float16)(batchNormalization)
    dropout = tf.keras.layers.Dropout(dropout_rate, weights=model.layers[4].get_weights(), dtype=tf.float16)(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = tf.keras.layers.Conv2D(80, (5, 5), strides=(3, 3), kernel_initializer="glorot_uniform",
                           weights=model.layers[5].get_weights(), dtype=tf.float16)(dropout)
    batchNormalization = tf.keras.layers.BatchNormalization(weights=model.layers[6].get_weights(), dtype=tf.float16)(conv2d)
    activationLayer = tf.keras.layers.Activation(activation, weights=model.layers[7].get_weights(), dtype=tf.float16)(batchNormalization)
    dropout = tf.keras.layers.Dropout(dropout_rate, weights=model.layers[8].get_weights(), dtype=tf.float16)(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = tf.keras.layers.Conv2D(80, (3, 3), strides=(2, 2), kernel_initializer="glorot_uniform",
                           weights=model.layers[9].get_weights(), dtype=tf.float16)(dropout)
    batchNormalization = tf.keras.layers.BatchNormalization(weights=model.layers[10].get_weights(), dtype=tf.float16)(conv2d)
    activationLayer = tf.keras.layers.Activation(activation, weights=model.layers[11].get_weights(), dtype=tf.float16)(batchNormalization)
    dropout = tf.keras.layers.Dropout(dropout_rate, weights=model.layers[12].get_weights(), dtype=tf.float16)(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = tf.keras.layers.Conv2D(80, (2, 2), strides=(1, 1), kernel_initializer="glorot_uniform",
                           weights=model.layers[13].get_weights(), dtype=tf.float16)(dropout)
    batchNormalization = tf.keras.layers.BatchNormalization(weights=model.layers[14].get_weights(), dtype=tf.float16)(conv2d)
    activationLayer = tf.keras.layers.Activation(activation, weights=model.layers[15].get_weights(), dtype=tf.float16)(batchNormalization)
    dropout = tf.keras.layers.Dropout(dropout_rate, weights=model.layers[16].get_weights(), dtype=tf.float16)(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    return dropout


def special_model(WIDTH, HEIGHT, unlock_segment_weights):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3))
    model1 = tf.keras.models.load_model("./segmented_models_3/color")
    model2 = tf.keras.models.load_model("./segmented_models_3/framing")
    model3 = tf.keras.models.load_model("./segmented_models_3/lighting")
    model4 = tf.keras.models.load_model("./segmented_models_3/symmetry")

    model = tf.keras.models.Sequential()

    m1 = build_layers_for_model(model1, input, unlock_segment_weights)
    m2 = build_layers_for_model(model2, input, unlock_segment_weights)
    m3 = build_layers_for_model(model3, input, unlock_segment_weights)
    m4 = build_layers_for_model(model4, input, unlock_segment_weights)

    do0 = new_conv2d(input, 80, (20, 20), (5, 5))
    do0 = new_conv2d(do0, 80, (5, 5), (3, 3))
    do0 = new_conv2d(do0, 80, (3, 3), (2, 2))
    do0 = new_conv2d(do0, 80, (2, 2), (1, 1))

    f1 = tf.keras.layers.Flatten()(do0)

    f1 = tf.keras.layers.Concatenate()(
            [tf.keras.layers.Flatten()(m1), tf.keras.layers.Flatten()(m2), tf.keras.layers.Flatten()(m3), tf.keras.layers.Flatten()(m4)])

    do1 = new_dense(f1, 256, dropout_rate=0.2)
    do2 = new_dense(do1, 128, dropout_rate=0.2)
    do2 = new_dense(do1, 64, dropout_rate=0.2)
    do2 = new_dense(do2, 10, dropout_rate=0.2)
    d4 = tf.keras.layers.Dense(1, kernel_initializer="he_uniform", activation="linear")(do2)
    model = tf.keras.models.Model(inputs=input, outputs=d4)

    return model


def build_layers_for_model2(model, input, unlock_segment_weights):
    activation = "relu"
    dropout_rate = 0.2
    conv2d = tf.keras.layers.Conv2D(80, (20, 20), strides=(5, 5), kernel_initializer="glorot_uniform",
                           weights=model.layers[1].get_weights(), dtype=tf.float16)(input)
    batchNormalization = tf.keras.layers.BatchNormalization(weights=model.layers[2].get_weights(), dtype=tf.float16)(conv2d)
    activationLayer = tf.keras.layers.Activation(activation, weights=model.layers[3].get_weights(), dtype=tf.float16)(batchNormalization)
    dropout = tf.keras.layers.Dropout(dropout_rate, weights=model.layers[4].get_weights(), dtype=tf.float16)(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = tf.keras.layers.Conv2D(80, (5, 5), strides=(3, 3), kernel_initializer="glorot_uniform",
                           weights=model.layers[5].get_weights(), dtype=tf.float16)(dropout)
    batchNormalization = tf.keras.layers.BatchNormalization(weights=model.layers[6].get_weights(), dtype=tf.float16)(conv2d)
    activationLayer = tf.keras.layers.Activation(activation, weights=model.layers[7].get_weights(), dtype=tf.float16)(batchNormalization)
    dropout = tf.keras.layers.Dropout(dropout_rate, weights=model.layers[8].get_weights(), dtype=tf.float16)(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = tf.keras.layers.Conv2D(80, (3, 3), strides=(2, 2), kernel_initializer="glorot_uniform",
                           weights=model.layers[9].get_weights(), dtype=tf.float16)(dropout)
    batchNormalization = tf.keras.layers.BatchNormalization(weights=model.layers[10].get_weights(), dtype=tf.float16)(conv2d)
    activationLayer = tf.keras.layers.Activation(activation, weights=model.layers[11].get_weights(), dtype=tf.float16)(batchNormalization)
    dropout = tf.keras.layers.Dropout(dropout_rate, weights=model.layers[12].get_weights(), dtype=tf.float16)(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = tf.keras.layers.Conv2D(80, (2, 2), strides=(1, 1), kernel_initializer="glorot_uniform",
                           weights=model.layers[13].get_weights(), dtype=tf.float16)(dropout)
    batchNormalization = tf.keras.layers.BatchNormalization(weights=model.layers[14].get_weights(), dtype=tf.float16)(conv2d)
    activationLayer = tf.keras.layers.Activation(activation, weights=model.layers[15].get_weights(), dtype=tf.float16)(batchNormalization)
    dropout = tf.keras.layers.Dropout(dropout_rate, weights=model.layers[16].get_weights(), dtype=tf.float16)(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    f1 = tf.keras.layers.Flatten()(dropout)

    dense = tf.keras.layers.Dense(64, activation=activation, kernel_initializer="glorot_uniform",
                         weights=model.layers[18].get_weights(), dtype=tf.float16)(f1)
    dropout = tf.keras.layers.Dropout(dropout_rate, weights=model.layers[19].get_weights(), dtype=tf.float16)(dense)
    dense = tf.keras.layers.Dense(64, activation=activation, kernel_initializer="glorot_uniform",
                         weights=model.layers[20].get_weights(), dtype=tf.float16)(dropout)
    dropout = tf.keras.layers.Dropout(dropout_rate, weights=model.layers[21].get_weights(), dtype=tf.float16)(dense)
    d4 = tf.keras.Dense(1, kernel_initializer="he_uniform", activation="hard_sigmoid", weights=model.layers[22].get_weights())(
        dropout)

    return d4


def special_model2(WIDTH, HEIGHT, unlock_segment_weights):
    input = tf.keras.layers.Input((WIDTH, HEIGHT, 3))
    model1 = tf.keras.models.load_model("./segmented_models_3/color")
    model2 = tf.keras.models.load_model("./segmented_models_3/framing")
    model3 = tf.keras.models.load_model("./segmented_models_3/lighting")
    model4 = tf.keras.models.load_model("./segmented_models_3/symmetry")

    model = tf.keras.models.Sequential()

    m1 = build_layers_for_model2(model1, input, unlock_segment_weights)
    m2 = build_layers_for_model2(model2, input, unlock_segment_weights)
    m3 = build_layers_for_model2(model3, input, unlock_segment_weights)
    m4 = build_layers_for_model2(model4, input, unlock_segment_weights)

    f1 = tf.keras.layers.Add()([m1, m2, m3, m4])

    model = tf.keras.models.Model(inputs=input, outputs=f1)

    return model