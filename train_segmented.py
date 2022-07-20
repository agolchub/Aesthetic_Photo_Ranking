#!/usr/bin/env python

import skimage
import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize
import sys, getopt
from tensorflow.keras import initializers, models, optimizers, layers
import os
import tensorflow as tf
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.lib.io.file_io import file_crc32
from tensorflow.python.platform.tf_logging import error
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.models import Sequential
from datetime import datetime
import matplotlib.pylab as plt
import itertools

class CustomDataGen(tf.keras.utils.Sequence):

    def __init__(self,
                 batch_size, path,
                 input_size=(1024, 680, 3),
                 shuffle=True, resnet=False):

        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.path = path
        self.items = glob(os.path.join(self.path, "*.jpg"))
        self.n = len(self.items)
        self.resnet = resnet

    def on_epoch_end(self):
        pass

    def __getitem__(self, index):
        # import random

        x = []  # images as arrays
        y = []  # labels Infiltration or Not_infiltration
        WIDTH = 1024
        HEIGHT = 680
        if self.resnet:
            WIDTH = HEIGHT = 224
        j = 0
        images = []
        rawscore = 0.0

        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            if (i >= len(self.items)):
                break
            item = self.items[i]
            # print("Reading " + item)
            # Read and resize image
            full_size_image = io.imread(item)
            rawscore = float(os.path.basename(item).split("-")[0])
            out = rawscore
            y.append(out)
            images.append(item)
            resizedImage = resize(full_size_image, (WIDTH, HEIGHT), anti_aliasing=True)
            if (len(resizedImage.shape) < 3):
                resizedImage = skimage.color.gray2rgb(resizedImage)

            x.append(resizedImage)
        x = np.array(x)
        y = np.array(y)
        if self.resnet:
            x = tf.keras.applications.resnet.preprocess_input(x)

        return x, y

    def __len__(self):
        return self.n // self.batch_size


class WaitCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        import time

        stream = os.popen("nvidia-smi -q -i 0 -d TEMPERATURE|grep 'GPU Current'|cut -d\" \" -f 30")
        temp = int(stream.read())
        print("The temperature of the GPU is ", temp)

        if (temp > 75):
            print("Cooling down...")
            while (temp > 55):
                time.sleep(15)
                stream = os.popen("nvidia-smi -q -i 0 -d TEMPERATURE|grep 'GPU Current'|cut -d\" \" -f 30")
                temp = int(stream.read())

            print("Resuming -> ")

        return super().on_epoch_end(epoch, logs=logs)


def proc_image_dir(Images_Path, scores="", categorical=False, WIDTH=1024, HEIGHT=680, scoreColumn=5):
    import random
    import csv
    import os
    import psutil
    import gc
    #    image_classes = sorted([dirname for dirname in os.listdir(Images_Path)
    #                      if os.path.isdir(os.path.join(Images_Path, dirname)) and not dirname.startswith(".") and not dirname.startswith("mblur")])

    #    print(image_classes)

    x = []  # images as arrays
    y = []  # labels Infiltration or Not_infiltration

    print("Adding Images: ", end="")
    i = 0
    #    for image_class in image_classes:
    # print("Processing ", image_class)
    items = glob(os.path.join(Images_Path, "*"))
    j = 0
    images = []
    rawscore = 0.0
    random.shuffle(items)
    # items = items[:200]

    if scores != "":
        with open(scores, mode='r') as cat:
            csvFile = csv.reader(cat)
            for line in csvFile:
                try:
                    resizedImage, out = read_one_image(HEIGHT, Images_Path, WIDTH, categorical, line, scoreColumn)
                    y.append(out)
                    x.append(resizedImage)
                    images.append(Images_Path + line[0])
                    gc.collect()
                    print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                except Exception as e:
                    print("Error ---- ")
                    print(e)
                    print(line)
        return x, y, images

    for item in items:
        print("Reading " + item)
        if item.lower().endswith(".jpg") or item.lower().endswith(".bmp"):
            # Read and resize image
            full_size_image = io.imread(item)

            rawscore = float(os.path.basename(item).split("-")[0])

            if rawscore >= 0:
                # x.append(full_size_image)
                if (categorical):
                    out = [0] * 5
                    out[int(rawscore)] = 1
                else:
                    out = rawscore  # ((rawscore - 1.0)/9.0)
                print(out)
                y.append(out)
                images.append(item)
                resizedImage = resize(full_size_image, (WIDTH, HEIGHT), anti_aliasing=True)
                if (len(resizedImage.shape) < 3):
                    resizedImage = skimage.color.gray2rgb(resizedImage)
                resizedImage = tf.keras.applications.resnet.preprocess_input(resizedImage)
                x.append(resizedImage)
                j += 1
            # if j>3:
            #    break

    print("\nRead " + str(j) + " images.\n\n")
    return x, y, images


def read_one_image(HEIGHT, Images_Path, WIDTH, categorical, line, scoreColumn):
    imagePath = Images_Path + line[0]
    full_size_image = io.imread(imagePath)
    resizedImage = resize(full_size_image, (WIDTH, HEIGHT), anti_aliasing=True)
    rawscore = int(line[scoreColumn])
    if resizedImage.shape[2] == 3:
        if (categorical):
            out = [0] * 5
            out[rawscore] = 1
        else:
            out = rawscore  # ((rawscore - 1.0)/9.0)

        print(line[scoreColumn] + " - " + imagePath)
    del full_size_image
    del rawscore
    return resizedImage, out

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
    conv2d = layers.Conv2D(n, size, strides=strides, kernel_initializer=kernel_initializer, padding=padding)(input)
    batch_normalization_layer = layers.BatchNormalization()(conv2d) if batch_normalization else conv2d
    activation_layer = layers.Activation(activation)(batch_normalization_layer) if activation is not None else batch_normalization_layer
    dropout = layers.Dropout(dropout_rate)(activation_layer) if dropout_rate > 0 else activation_layer

    return dropout


def new_dense(input, n, activation="relu", kernel_initializer="glorot_uniform", dropout_rate=0.2):
    dense = layers.Dense(n, activation=activation, kernel_initializer=kernel_initializer)(input)
    dropout = layers.Dropout(dropout_rate)(dense)
    return dropout


def new_res_block(input, n, m, size, strides):
    conv2d = layers.Conv2D(n, size, strides=strides)(input)
    batchNormalization = layers.BatchNormalization()(conv2d)
    activation = layers.Activation("relu")(batchNormalization)
    shortcut = activation
    conv2d = layers.Conv2D(n * m, (2, 2), strides=(1, 1))(activation)
    batchNormalization = layers.BatchNormalization()(conv2d)
    activation = layers.Activation("relu")(batchNormalization)
    conv2d = layers.Conv2D(n * m * m, (2, 2), strides=(1, 1))(activation)
    batchNormalization = layers.BatchNormalization()(conv2d)

    shortcut = layers.Conv2D(n * m * m, (3, 3), strides=(1, 1))(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    add = layers.Add()([batchNormalization, shortcut])
    activation = layers.Activation("relu")(add)
    return activation


def new_res_block_v2(input, n, size=3, strides=1, first_strides=1):
    shortcut = input
    if (first_strides != strides):
        shortcut = layers.Conv2D(n, (1, 1), strides=first_strides, padding='same', activation=None)(shortcut)
    conv2d = layers.Conv2D(n, size, strides=first_strides, padding='same', activation=None)(input)
    batch_normalization = layers.BatchNormalization()(conv2d)
    activation = layers.Activation("relu")(batch_normalization)
    conv2d = layers.Conv2D(n, size, strides=strides, padding='same', activation=None)(activation)
    batch_normalization = layers.BatchNormalization()(conv2d)
    add = layers.Add()([batch_normalization, shortcut])
    activation = layers.Activation("relu")(add)
    return activation


def new_res_block_collection_v2(blocks, input, n, size=3, strides=1, first_strides=1):
    activation = input
    for i in range(blocks):
        print(i)
        activation = new_res_block_v2(activation, n, size, strides, strides if i != 0 else first_strides)
    return activation


def build_layers_for_model(model, input, unlock_segment_weights):
    activation = "relu"
    dropout_rate = 0.2
    conv2d = layers.Conv2D(80, (20, 20), strides=(5, 5), kernel_initializer="glorot_uniform",
                           weights=model.layers[1].get_weights())(input)
    batchNormalization = layers.BatchNormalization(weights=model.layers[2].get_weights())(conv2d)
    activationLayer = layers.Activation(activation, weights=model.layers[3].get_weights())(batchNormalization)
    dropout = layers.Dropout(dropout_rate, weights=model.layers[4].get_weights())(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = layers.Conv2D(80, (5, 5), strides=(3, 3), kernel_initializer="glorot_uniform",
                           weights=model.layers[5].get_weights())(dropout)
    batchNormalization = layers.BatchNormalization(weights=model.layers[6].get_weights())(conv2d)
    activationLayer = layers.Activation(activation, weights=model.layers[7].get_weights())(batchNormalization)
    dropout = layers.Dropout(dropout_rate, weights=model.layers[8].get_weights())(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = layers.Conv2D(80, (3, 3), strides=(2, 2), kernel_initializer="glorot_uniform",
                           weights=model.layers[9].get_weights())(dropout)
    batchNormalization = layers.BatchNormalization(weights=model.layers[10].get_weights())(conv2d)
    activationLayer = layers.Activation(activation, weights=model.layers[11].get_weights())(batchNormalization)
    dropout = layers.Dropout(dropout_rate, weights=model.layers[12].get_weights())(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = layers.Conv2D(80, (2, 2), strides=(1, 1), kernel_initializer="glorot_uniform",
                           weights=model.layers[13].get_weights())(dropout)
    batchNormalization = layers.BatchNormalization(weights=model.layers[14].get_weights())(conv2d)
    activationLayer = layers.Activation(activation, weights=model.layers[15].get_weights())(batchNormalization)
    dropout = layers.Dropout(dropout_rate, weights=model.layers[16].get_weights())(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    return dropout


def build_layers_for_model2(model, input, unlock_segment_weights):
    activation = "relu"
    dropout_rate = 0.2
    conv2d = layers.Conv2D(80, (20, 20), strides=(5, 5), kernel_initializer="glorot_uniform",
                           weights=model.layers[1].get_weights())(input)
    batchNormalization = layers.BatchNormalization(weights=model.layers[2].get_weights())(conv2d)
    activationLayer = layers.Activation(activation, weights=model.layers[3].get_weights())(batchNormalization)
    dropout = layers.Dropout(dropout_rate, weights=model.layers[4].get_weights())(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = layers.Conv2D(80, (5, 5), strides=(3, 3), kernel_initializer="glorot_uniform",
                           weights=model.layers[5].get_weights())(dropout)
    batchNormalization = layers.BatchNormalization(weights=model.layers[6].get_weights())(conv2d)
    activationLayer = layers.Activation(activation, weights=model.layers[7].get_weights())(batchNormalization)
    dropout = layers.Dropout(dropout_rate, weights=model.layers[8].get_weights())(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = layers.Conv2D(80, (3, 3), strides=(2, 2), kernel_initializer="glorot_uniform",
                           weights=model.layers[9].get_weights())(dropout)
    batchNormalization = layers.BatchNormalization(weights=model.layers[10].get_weights())(conv2d)
    activationLayer = layers.Activation(activation, weights=model.layers[11].get_weights())(batchNormalization)
    dropout = layers.Dropout(dropout_rate, weights=model.layers[12].get_weights())(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    conv2d = layers.Conv2D(80, (2, 2), strides=(1, 1), kernel_initializer="glorot_uniform",
                           weights=model.layers[13].get_weights())(dropout)
    batchNormalization = layers.BatchNormalization(weights=model.layers[14].get_weights())(conv2d)
    activationLayer = layers.Activation(activation, weights=model.layers[15].get_weights())(batchNormalization)
    dropout = layers.Dropout(dropout_rate, weights=model.layers[16].get_weights())(activationLayer)
    conv2d.trainable = unlock_segment_weights
    batchNormalization.trainable = unlock_segment_weights
    activationLayer.trainable = unlock_segment_weights
    dropout.trainable = unlock_segment_weights

    f1 = layers.Flatten()(dropout)

    dense = layers.Dense(64, activation=activation, kernel_initializer="glorot_uniform",
                         weights=model.layers[18].get_weights())(f1)
    dropout = layers.Dropout(dropout_rate, weights=model.layers[19].get_weights())(dense)
    dense = layers.Dense(64, activation=activation, kernel_initializer="glorot_uniform",
                         weights=model.layers[20].get_weights())(dropout)
    dropout = layers.Dropout(dropout_rate, weights=model.layers[21].get_weights())(dense)
    d4 = Dense(1, kernel_initializer="he_uniform", activation="hard_sigmoid", weights=model.layers[22].get_weights())(
        dropout)

    return d4


def train(modelin, modelout, imagepath, epochs, batch_size, lr, decay, nesterov, checkpoint_filepath, train_path,
          val_path, transfer_learning, randomize_weights, use_resnet, special_model, build_only, special_model2,
          batched_reader, simple_model, momentum, loss_function, catalog, WIDTH, HEIGHT, outColumn,
          unlock_segment_weights, model_design, reload, patience):
    categorical = False
    # load model
    if (simple_model):
        input = layers.Input((WIDTH, HEIGHT, 3))
        do0 = new_conv2d(input, 96, (7, 7), (30, 30))
        # do0 = new_conv2d(do0,256,(5,5),(3,3))
        # do0 = new_conv2d(do0,384,(3,3),(2,2))
        # do0 = new_conv2d(do0,384,(2,2),(2,2))
        # do0 = new_conv2d(do0, 256, (2, 2), (2, 2))

        f1 = layers.Flatten()(do0)

        do1 = new_dense(f1, 4096, dropout_rate=0.2)
        do2 = new_dense(do1, 4096, dropout_rate=0.2)
        do2 = new_dense(do2, 1024, dropout_rate=0.2)
        d4 = Dense(1, kernel_initializer="he_uniform", activation="sigmoid")(do2)
        model = models.Model(inputs=input, outputs=d4)
    elif (special_model):
        input = layers.Input((WIDTH, HEIGHT, 3))
        model1 = models.load_model("./segmented_models_3/color")
        model2 = models.load_model("./segmented_models_3/framing")
        model3 = models.load_model("./segmented_models_3/lighting")
        model4 = models.load_model("./segmented_models_3/symmetry")

        model = models.Sequential()

        m1 = build_layers_for_model(model1, input, unlock_segment_weights)
        m2 = build_layers_for_model(model2, input, unlock_segment_weights)
        m3 = build_layers_for_model(model3, input, unlock_segment_weights)
        m4 = build_layers_for_model(model4, input, unlock_segment_weights)

        do0 = new_conv2d(input, 80, (20, 20), (5, 5))
        do0 = new_conv2d(do0, 80, (5, 5), (3, 3))
        do0 = new_conv2d(do0, 80, (3, 3), (2, 2))
        do0 = new_conv2d(do0, 80, (2, 2), (1, 1))

        f1 = layers.Flatten()(do0)

        f1 = layers.Concatenate()(
            [layers.Flatten()(m1), layers.Flatten()(m2), layers.Flatten()(m3), layers.Flatten()(m4)])

        do1 = new_dense(f1, 256, dropout_rate=0.2)
        do2 = new_dense(do1, 128, dropout_rate=0.2)
        do2 = new_dense(do1, 64, dropout_rate=0.2)
        do2 = new_dense(do2, 10, dropout_rate=0.2)
        d4 = Dense(1, kernel_initializer="he_uniform", activation="linear")(do2)
        model = models.Model(inputs=input, outputs=d4)
    elif (special_model2):
        input = layers.Input((WIDTH, HEIGHT, 3))
        model1 = models.load_model("./segmented_models_3/color")
        model2 = models.load_model("./segmented_models_3/framing")
        model3 = models.load_model("./segmented_models_3/lighting")
        model4 = models.load_model("./segmented_models_3/symmetry")

        model = models.Sequential()

        m1 = build_layers_for_model2(model1, input, unlock_segment_weights)
        m2 = build_layers_for_model2(model2, input, unlock_segment_weights)
        m3 = build_layers_for_model2(model3, input, unlock_segment_weights)
        m4 = build_layers_for_model2(model4, input, unlock_segment_weights)

        f1 = layers.Add()([m1, m2, m3, m4])

        model = models.Model(inputs=input, outputs=f1)
    elif model_design == 3:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
        maxpool = layers.MaxPooling2D()(conv2d)
        res = new_res_block_collection_v2(3, maxpool, 64)
        res = new_res_block_collection_v2(4, res, 128, first_strides=(2, 2))
        res = new_res_block_collection_v2(6, res, 256, first_strides=2)
        res = new_res_block_collection_v2(3, res, 512, first_strides=2)
        flat = layers.Flatten()(res)
        # dense = new_dense(flat, 4096)
        # dense = new_dense(dense, 2048)
        # dense = new_dense(dense, 1024)
        # dense = new_dense(dense, 512)
        # dense = new_dense(dense, 256)
        # dense = new_dense(dense, 128)
        dense = Dense(5, kernel_initializer="he_uniform", activation="softmax")(flat)
        model = models.Model(inputs=input, outputs=dense)

    elif model_design == 4:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
        maxpool = layers.MaxPooling2D()(conv2d)
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

        flat1 = layers.Flatten()(res1)
        flat2 = layers.Flatten()(res2)
        flat3 = layers.Flatten()(res3)
        flat4 = layers.Flatten()(res4)

        dense1 = new_dense(flat1, 16)
        dense2 = new_dense(flat2, 16)
        dense3 = new_dense(flat3, 16)
        dense4 = new_dense(flat4, 16)

        concat = Concatenate()([dense1, dense2, dense3, dense4])

        output = Dense(1, kernel_initializer="he_uniform", activation="linear")(concat)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 5:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
        maxpool = layers.MaxPooling2D()(conv2d)
        res1 = new_res_block_collection_v2(3, maxpool, 64)
        res2 = new_res_block_collection_v2(4, res1, 128, first_strides=(2, 2))
        res3 = new_res_block_collection_v2(6, res2, 256, first_strides=2)
        res4 = new_res_block_collection_v2(3, res3, 512, first_strides=2)

        res1 = new_conv2d(res1, 128, (3, 3), strides=(2, 2), padding="same")
        res1 = new_conv2d(res1, 256, (2, 2), strides=(2, 2), padding="same")
        res1 = new_conv2d(res1, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

        res2 = new_conv2d(res2, 256, (3, 3), strides=(2, 2), padding="same")
        res2 = new_conv2d(res2, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

        res3 = new_conv2d(res3, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

        add = layers.Add()([res1, res2, res3, res4])
        res_combined = layers.Activation("relu")(add)

        flat = layers.Flatten()(res_combined)

        output = Dense(1, kernel_initializer="he_uniform", activation="linear")(flat)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 6:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
        maxpool = layers.MaxPooling2D()(conv2d)
        res1 = new_res_block_collection_v2(3, maxpool, 64)
        res2 = new_res_block_collection_v2(4, res1, 128, first_strides=(2, 2))
        res3 = new_res_block_collection_v2(6, res2, 256, first_strides=2)
        res4 = new_res_block_collection_v2(3, res3, 512, first_strides=2)

        res1 = new_conv2d(res1, 128, (3, 3), strides=(2, 2), padding="same")
        res1 = new_conv2d(res1, 256, (2, 2), strides=(2, 2), padding="same")
        res1 = new_conv2d(res1, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

        res2 = new_conv2d(res2, 256, (3, 3), strides=(2, 2), padding="same")
        res2 = new_conv2d(res2, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

        res3 = new_conv2d(res3, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

        add = layers.Add()([res1, res2, res3, res4])
        res_combined = layers.Activation("relu")(add)

        flat = layers.Flatten()(res_combined)

        dense = new_dense(flat, 128)

        output = Dense(1, kernel_initializer="he_uniform", activation="linear")(dense)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 7:
        input = layers.Input((WIDTH, HEIGHT, 3))
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
        #conv2d_2 = new_conv2d(conv2d, 512, (2, 2), strides=(1, 1), padding="same", batch_normalization=False)

        conv2d = new_conv2d(input, 64, (7, 7), strides=(2, 2), padding="same")
        conv2d = new_conv2d(conv2d, 128, (5, 5), strides=(2, 2), padding="same")
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(2, 2), padding="same")
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")
        conv2d_3 = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), padding="same")

        concat = layers.Concatenate()([conv2d_1, conv2d_2, conv2d_3])
        res_combined = layers.Activation("relu")(concat)

        flat = layers.Flatten()(res_combined)

        dense = Dense(128)(flat)
        output = Dense(1, kernel_initializer="he_uniform", activation="linear")(dense)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 8:
        input = layers.Input((WIDTH, HEIGHT, 3))
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

        flat_1 = layers.Flatten()(conv2d_1)
        flat_2 = layers.Flatten()(conv2d_2)
        flat_3 = layers.Flatten()(conv2d_3)
        flat_4 = layers.Flatten()(conv2d_4)

        concat = layers.Concatenate()([flat_1, flat_2, flat_3, flat_4])

        output = Dense(1, kernel_initializer="he_uniform", activation="linear")(concat)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 9:
        input = layers.Input((WIDTH, HEIGHT, 3))
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

        flat_1 = layers.Flatten()(conv2d_1)
        flat_2 = layers.Flatten()(conv2d_2)
        flat_3 = layers.Flatten()(conv2d_3)
        flat_4 = layers.Flatten()(conv2d_4)

        concat = layers.Concatenate()([flat_1, flat_2, flat_3, flat_4])

        output = Dense(5, kernel_initializer="he_uniform", activation="softmax")(concat)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 10:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
        maxpool = layers.MaxPooling2D()(conv2d)
        res1 = new_res_block_collection_v2(3, maxpool, 64)
        res2 = new_res_block_collection_v2(4, res1, 128, first_strides=(2, 2))
        res3 = new_res_block_collection_v2(6, res2, 256, first_strides=2)
        res4 = new_res_block_collection_v2(3, res3, 512, first_strides=2)

        res1 = new_conv2d(res1, 128, (3, 3), strides=(2, 2), padding="same")
        res1 = new_conv2d(res1, 256, (2, 2), strides=(2, 2), padding="same")
        res1 = new_conv2d(res1, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

        res2 = new_conv2d(res2, 256, (3, 3), strides=(2, 2), padding="same")
        res2 = new_conv2d(res2, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

        res3 = new_conv2d(res3, 512, (2, 2), strides=(2, 2), dropout_rate=0, padding="same", activation=None, batch_normalization=False)

        add = layers.Add()([res1, res2, res3, res4])
        res_combined = layers.Activation("relu")(add)

        flat = layers.Flatten()(res_combined)

        output = Dense(5, kernel_initializer="he_uniform", activation="softmax")(flat)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 11:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
        maxpool = layers.MaxPooling2D()(conv2d)
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

        flat1 = layers.Flatten()(res1)
        flat2 = layers.Flatten()(res2)
        flat3 = layers.Flatten()(res3)
        flat4 = layers.Flatten()(res4)

        dense1 = new_dense(flat1, 16)
        dense2 = new_dense(flat2, 16)
        dense3 = new_dense(flat3, 16)
        dense4 = new_dense(flat4, 16)

        concat = Concatenate()([dense1, dense2, dense3, dense4])

        output = Dense(5, kernel_initializer="he_uniform", activation="softmax")(concat)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 12:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 80, (102, 68), strides=(3, 2))
        conv2d = new_conv2d(conv2d, 128, (7, 7), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(1, 1))
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(1, 1))
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(1, 1))
        conv2d = new_conv2d(conv2d, 128, (3, 3), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 128, (3, 3), strides=(1, 1))

        flat = Flatten()(conv2d)

        dense = Dense(128, activation="sigmoid")(flat)
        dense = Dense(128, activation="sigmoid")(dense)

        output = Dense(5, kernel_initializer="he_uniform", activation="softmax")(dense)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 13:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 80, (12, 8), strides=(6, 4))
        conv2d = new_conv2d(conv2d, 128, (7, 7), strides=(4, 4))
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(4, 4))
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(4, 4))

        flat = Flatten()(conv2d)

        dense = Dropout(0.2)(Dense(128, activation="relu")(flat))
        dense = Dropout(0.2)(Dense(128, activation="relu")(dense))

        output = Dense(5, kernel_initializer="he_uniform", activation="softmax")(dense)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 14:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 80, (12, 8), strides=(3, 2))
        conv2d = new_conv2d(conv2d, 128, (7, 7), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2))

        flat = Flatten()(conv2d)

        conv2d_2 = new_conv2d(input, 80, (24, 16), strides=(6, 4))
        conv2d_2 = new_conv2d(conv2d_2, 128, (14, 14), strides=(2, 2))
        conv2d_2 = new_conv2d(conv2d_2, 256, (5, 5), strides=(2, 2))
        conv2d_2 = new_conv2d(conv2d_2, 512, (3, 3), strides=(2, 2))
        conv2d_2 = new_conv2d(conv2d_2, 512, (3, 3), strides=(2, 2))
        conv2d_2 = new_conv2d(conv2d_2, 512, (3, 3), strides=(2, 2))


        flat_2 = Flatten()(conv2d_2)
        concat = Concatenate()([flat, flat_2])

        dense = Dropout(0.2)(Dense(128, activation="relu")(concat))
        dense = Dropout(0.2)(Dense(128, activation="relu")(dense))

        output = Dense(5, kernel_initializer="he_uniform", activation="softmax")(dense)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 15:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
        maxpool = layers.MaxPooling2D()(conv2d)
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

        add = layers.Add()([res1, res2, res3, res4])
        res_combined = layers.Activation("relu")(add)

        flat = layers.Flatten()(res_combined)

        output = Dense(5, kernel_initializer="he_uniform", activation="linear")(flat)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 16:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
        maxpool = layers.MaxPooling2D()(conv2d)
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

        add = layers.Add()([res1, res2, res3, res4])
        res_combined = layers.Activation("relu")(add)

        conv2d = new_conv2d(res_combined, 512, (3, 3), strides=(3, 2), dropout_rate=0.2)
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), dropout_rate=0.2)

        flat = layers.Flatten()(conv2d)

        output = Dense(5, kernel_initializer="he_uniform", activation="linear")(flat)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 17:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 80, (12, 8), strides=(6, 4))
        conv2d = new_conv2d(conv2d, 128, (7, 7), strides=(4, 4))
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(4, 4))
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(4, 4))

        flat = Flatten()(conv2d)

        conv2d = new_conv2d(input, 80, (12, 8), strides=(3, 2))
        conv2d = new_conv2d(conv2d, 128, (7, 7), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 256, (3, 3), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2))
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2))

        concat = Concatenate()([Flatten()(conv2d), flat])

        dense = Dropout(0.2)(Dense(128, activation="relu")(concat))
        dense = Dropout(0.2)(Dense(128, activation="relu")(dense))

        output = Dense(5, kernel_initializer="he_uniform", activation="softmax")(dense)
        model = models.Model(inputs=input, outputs=output)

    elif model_design == 18:
        input = layers.Input((WIDTH, HEIGHT, 3))
        conv2d = new_conv2d(input, 64, (7, 7), strides=(1, 1))
        maxpool = layers.MaxPooling2D()(conv2d)
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

        add = layers.Add()([res1, res2, res3, res4])
        res_combined = layers.Activation("relu")(add)

        conv2d = new_conv2d(res_combined, 512, (3, 3), strides=(3, 2), dropout_rate=0.2)
        conv2d = new_conv2d(conv2d, 512, (3, 3), strides=(2, 2), dropout_rate=0.2)

        flat = layers.Flatten()(conv2d)

        output = Dense(5, kernel_initializer="he_uniform", activation="softmax")(flat)
        model = models.Model(inputs=input, outputs=output)

    else:
        model = models.load_model(modelin)


    categorical = model.output_shape[1] > 1
    print("categorical: " + str(categorical))

    if transfer_learning:
        for layer in model.layers:
            if (isinstance(layer, tf.keras.layers.Conv2D)):
                print("conv layer", layer.name)
                layer.trainable = False

    if randomize_weights:

        for layer in model.layers:
            if (layer.trainable):
                init_layer(layer)

    # Define the Keras TensorBoard callback.
    logdir = modelout + ".logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # model.build()
    model.compile(
        loss='mse',
        optimizer=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

    print(model.summary())
    if (build_only):
        img_file = modelout + '-model_arch.png'
        tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)
        model.save(modelout)
        exit(0)

    if (imagepath != ''):
        # load images
        x2, y2, images = proc_image_dir(imagepath)

        # First split the data in two sets, 60% for training, 40% for Val/Test)
        X_train, X_valtest, y_train, y_valtest = train_test_split(x2, y2, test_size=0.4, random_state=1)

        # Second split the 40% into validation and test sets
        X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1)
    elif (batched_reader):
        training_generator = CustomDataGen(batch_size, train_path, resnet=use_resnet)
        validation_generator = CustomDataGen(batch_size, val_path, resnet=use_resnet)
    else:
        X_train, y_train, image_list_train = proc_image_dir(os.path.dirname(os.path.abspath(train_path)) + '/JPEG/',
                                                            train_path, WIDTH=WIDTH, HEIGHT=HEIGHT,
                                                            scoreColumn=outColumn, categorical=categorical)
        X_val, y_val, image_list_val = proc_image_dir(os.path.dirname(os.path.abspath(val_path)) + '/JPEG/', val_path,
                                                      WIDTH=WIDTH, HEIGHT=HEIGHT, scoreColumn=outColumn, categorical=categorical)

    print("Images loaded")

    # run training loop
    wait_callback = WaitCallback()

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1,
                                                               restore_best_weights=reload)

    steps_per_epoch = int(len(X_train)/batch_size)
    epochs_per_rate = int((epochs / len(lr)))



    for l in lr:
        learning_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=l,
                                                                  decay_steps=steps_per_epoch, decay_rate=1.0-decay)
        print(reload)
        print(lr.index(l))
        if reload and lr.index(l) > 0:
            print("reloading checkpoint weights...")
            model.load_weights(checkpoint_filepath)

        print("learning rate: " + str(l))
        model.compile(
            loss=loss_function,
            optimizer=optimizers.SGD(learning_rate=learning_schedule, momentum=momentum, nesterov=nesterov),
            metrics=['accuracy'])

        # run training loop
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        wait_callback = WaitCallback()


        # model.build()
        # history = model.fit(np.array(X_train), np.array(y_train),
        #    validation_data=(np.array(X_val), np.array(y_val)),
        #        epochs=epochs, batch_size=batch_size,
        #        callbacks=[model_checkpoint_callback,wait_callback])
        if (batched_reader):
            history = model.fit_generator(generator=training_generator,
                                          validation_data=validation_generator,
                                          epochs=epochs_per_rate,
                                          callbacks=[model_checkpoint_callback])
        else:
            history = model.fit(np.array(X_train), np.array(y_train),
                                validation_data=(np.array(X_val), np.array(y_val)),
                                epochs=epochs_per_rate, batch_size=batch_size,
                                callbacks=[model_checkpoint_callback,
                                           tensorboard_callback, early_stopping_callback])

        # save model
        model.save(modelout + "/model")

        print([model.history.history["loss"], model.history.history["val_loss"]])

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figure_size=(5,5)):
    from matplotlib.transforms import offset_copy

    plt.figure(figsize = figure_size)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    y_tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(y_tick_marks,labels=classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("./cm-"+title+".png")
    plt.show()

def test(modelin, imagepath, WIDTH, HEIGHT, outColumn, weights = None):
    from sklearn.metrics import confusion_matrix
    model = models.load_model(modelin)
    # model.load_weights(modelin+".checkpoint/")
    print("Model Loaded")
    print(model.summary())

    if weights is not None:
        print("Loading weights...")
        model.load_weights(weights)
        print("Weights loaded")

    categorical = model.output_shape[1] > 1

    x, y, images = proc_image_dir(os.path.dirname(os.path.abspath(imagepath)) + '/JPEG/', imagepath, WIDTH=WIDTH,
                                  HEIGHT=HEIGHT, scoreColumn=outColumn, categorical=categorical)
    a = np.array(x).astype(float)
    Y_pred = model.predict(a)
    print(np.array(y))
    print(Y_pred)

    np.set_printoptions(threshold=sys.maxsize)

    print(np.argmax(np.array(y), axis=1))
    print(np.argmax(Y_pred, axis=1))

    loss, acc = model.evaluate(a, np.array(y), verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    Y_pred_classes = np.argmax(Y_pred, axis=1)
    confusion_mtx = confusion_matrix(np.argmax(np.array(y), axis=1), Y_pred_classes)
    plot_confusion_matrix(confusion_mtx, classes=[0, 1, 2, 3, 4], title='Confusion Matrix - Test Set')


def main(argv):
    modelin = ''
    modelout = ''
    imagepath = ''
    epochs = 1
    batch_size = 1
    lr = []
    decay = 0.0
    nesterov = False
    checkpoint_filepath = "./checkpoint/"
    train_path = ''
    val_path = ''
    transfer_learning = False
    randomize_weights = False
    use_resnet = False
    special_model = False
    build_only = False
    testmode = False
    special_model2 = False
    batched_reader = False
    simple_model = False
    catalog = ""
    momentum = 0.0
    loss_function = 'mse'
    WIDTH = 1024
    HEIGHT = 680
    outColumn = 5
    unlock_segment_weights = False
    model_design = 0
    reload = False
    patience = 0
    weights = None

    try:
        opts, args = getopt.getopt(argv, "hi:o:p:nd:l:b:e:c:t:v:xrm:f:",
                                   ["unlock_segment_weights", "outColumn=", "width=", "height=", "catalog=", "loss=",
                                    "momentum=", "special_model2", "simple_model", "test", "build_only", "modelin=",
                                    "resnet50", "special_model", "modelout=", "imagepath=", "nesterov", "decay=",
                                    "learningrate=", "batchsize", "epochs", "checkpoint_filepath=", "train=", "val=",
                                    "test=", "transfer_learning", "randomize_weights", "batched_reader",
                                    "model_design=", "reload_checkpoint_between_rates", "patience=", "load_weights="])
    except getopt.GetoptError:
        print('train.py -i <modelin> -o <modelout> -p <imagepath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -i <modelin> -o <modelout> -p <imagepath>')
            sys.exit()
        elif opt in ("-i", "--modelin"):
            modelin = arg
        elif opt in ("-o", "--modelout"):
            modelout = arg
        elif opt in ("-p", "--imagepath"):
            imagepath = arg
        elif opt in ("-d", "--decay"):
            decay = float(arg)
        elif opt in ("-n", "--nesterov"):
            nesterov = True
        elif opt in ("-l", "--learningrate"):
            lr.append(float(arg))
        elif opt in ("-b", "--batchsize"):
            batch_size = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-t", "--train"):
            train_path = arg
        elif opt in ("-v", "--val"):
            val_path = arg
        elif opt in ("-x", "--transfer_learning"):
            transfer_learning = True
        elif opt in ("-r", "--randomize_weights"):
            randomize_weights = True
        elif opt in ("--resnet50"):
            use_resnet = True
        elif opt in ("--special_model"):
            special_model = True
        elif opt in ("--special_model2"):
            special_model2 = True
        elif opt in ("--build_only"):
            build_only = True
        elif opt in ("--test"):
            testmode = True
        elif opt in ("--batched_reader"):
            batched_reader = True
        elif opt in ("--simple_model"):
            simple_model = True
        elif opt in ("-e", "--momentum"):
            momentum = float(arg)
        elif opt in ("-f", "--loss"):
            loss_function = arg
        elif opt in ("--catalog"):
            catalog = arg
        elif opt in ("--width"):
            WIDTH = int(arg)
        elif opt in ("--height"):
            HEIGHT = int(arg)
        elif opt in ("--outColumn"):
            outColumn = int(arg)
        elif opt in ("--unlock_segment_weights"):
            unlock_segment_weights = True
        elif opt in ("--model_design"):
            model_design = int(arg)
        elif opt in ("--reload_checkpoint_between_rates"):
            reload = True
        elif opt in ("--patience"):
            patience = int(arg)
        elif opt in ("--load_weights"):
            weights = arg

    checkpoint_filepath = modelout + ".checkpoint/"

    if len(lr) < 1:
        lr.append(0.01)

    print('Input file is "', modelin)
    print('Output file is "', modelout)
    print('Image path is "', imagepath)

    if (not testmode and ((
                                  modelin == '' and not use_resnet and not special_model and not special_model2 and not simple_model and model_design < 3) or modelout == '' or (
                                  imagepath == '' and (train_path == '' or val_path == '')))):
        print('Missing required parameter.')
        print('train.py -i <modelin> -o <modelout> -p <imagepath>')
        sys.exit(2)

    if (testmode and (modelin == '' or imagepath == '')):
        print('Missing required parameter.')
        print('train.py --test -i <modelin> -p <imagepath>')
        sys.exit(2)

    print('--------------------\n\n')

    if (testmode):
        test(modelin, imagepath, WIDTH, HEIGHT, outColumn, weights)
    else:
        train(modelin, modelout, imagepath, epochs, batch_size, lr, decay, nesterov, checkpoint_filepath, train_path,
              val_path, transfer_learning, randomize_weights, use_resnet, special_model, build_only, special_model2,
              batched_reader, simple_model, momentum, loss_function, catalog, WIDTH, HEIGHT, outColumn,
              unlock_segment_weights, model_design, reload, patience)


if __name__ == "__main__":
    main(sys.argv[1:])
