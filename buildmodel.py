#!/usr/bin/env python

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import tensorflow as tf
import pickle
import math, sys, getopt

def main(argv):
    modelout = ''
    imagewidth = 0
    imageheight = 0

    try:
        opts, args = getopt.getopt(argv,"h:w:o:",["modelout=","inputwidth=","inputheight=","help"])
    except getopt.GetoptError:
        print ('Unrecognized option.\ntrain.py -o <modelout> -w <inputwidth> -h <inputheight>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '--help':
            print ('train.py -o <modelout> -w <inputwidth> -h <inputheight>')
            sys.exit()
        elif opt in ("-o", "--modelout"):
            modelout = arg
        elif opt in ("-w", "--inputwidth"):
            imagewidth = int(arg)
        elif opt in ("-h", "--inputheight"):
            imageheight = int(arg)       
            
    print ('Output file is "', modelout)
    print ('Input image size is "', imagewidth, ", ", imageheight)

    if(modelout == '' or imagewidth == 0 or imageheight == 0):
        print('Missing required parameter.')
        print ('train.py -o <modelout> -w <inputwidth> -h <inputheight>')
        sys.exit(2)

    print ('--------------------\n\n')

    K.image_data_format()

    img_width, img_height = imagewidth, imageheight

    initializer = tf.keras.initializers.GlorotNormal()

    model = models.Sequential()
    model.add(layers.Conv2D(32, (132, 88),input_shape=(img_width, img_height, 3), strides=(3,2), activation="relu", kernel_initializer="he_uniform"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (66, 66), strides=(2,2), activation="relu",kernel_initializer="he_uniform"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (16, 16), strides=(2,2), activation="relu",kernel_initializer="he_uniform"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(256, (7, 7), strides=(2,2), activation="relu",kernel_initializer="he_uniform"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(512, (3, 3), strides=(1,1), activation="relu",kernel_initializer="he_uniform"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())

    model.add(layers.Dense(256, activation="relu",kernel_initializer="he_uniform"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(128, activation="relu",kernel_initializer="he_uniform"))
    model.add(layers.Dropout(0.2))

    model.add(Dense(1, input_dim=1, kernel_initializer="he_uniform", activation="linear"))

    model.compile(
        loss='mse',
        optimizer=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

    model.summary() 

    run_filepath = modelout
    model.save(run_filepath)

if __name__ == "__main__":
    main(sys.argv[1:])