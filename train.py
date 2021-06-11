import skimage
import os
import random
import matplotlib.pylab as plt
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import rescale, resize
from datetime import datetime
import math, sys, getopt
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

def proc_image_dir(Images_Path):
    
#    image_classes = sorted([dirname for dirname in os.listdir(Images_Path)
#                      if os.path.isdir(os.path.join(Images_Path, dirname)) and not dirname.startswith(".") and not dirname.startswith("mblur")])
    
#    print(image_classes)
    
    x = [] # images as arrays
    y = [] # labels Infiltration or Not_infiltration
    WIDTH = 1024
    HEIGHT = 680
  
    print("Adding Images: ",end="")
    i = 0
#    for image_class in image_classes:
    #print("Processing ", image_class)
    items = glob(os.path.join(Images_Path,"*"))
    j = 0
    images = []
    rawscore = 0.0

    for item in items:
        print("Reading "+item)
        if item.lower().endswith(".jpg") or item.lower().endswith(".bmp"):
            # Read and resize image
            full_size_image = io.imread(item)

            rawscore = int(os.path.basename(item).split("-")[0])

            if rawscore >= 1:
                #x.append(full_size_image)
                out = rawscore
                print(out)
                y.append(out)
                images.append(item)
                resizedImage = resize(full_size_image, (WIDTH,HEIGHT), anti_aliasing=True) 
                if(len(resizedImage.shape) < 3):
                    resizedImage = skimage.color.gray2rgb(resizedImage)

                x.append(resizedImage)
                j+=1
            #if j>3:
            #    break
                

    print("\nRead " + str(j) + " images.\n\n")
    return x,y,images

def horizontal_motion_blur(img, blur_factor):
    import cv2 
    import numpy as np 

    kernel_size = blur_factor
    kernel_h = np.zeros((kernel_size, kernel_size))
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
    kernel_h /= kernel_size 

    # Apply the horizontal kernel. 
    horizontal_mb = cv2.filter2D(img, -1, kernel_h) 
    
    return horizontal_mb

def train(modelin,modelout,imagepath,epochs,batch_size,lr,decay,nesterov,checkpoint_filepath):
    #load images
    x2,y2,images = proc_image_dir(imagepath)
    
    #load model
    model = models.load_model(modelin)
    print(model.summary())

    # First split the data in two sets, 60% for training, 40% for Val/Test)
    X_train, X_valtest, y_train, y_valtest = train_test_split(x2,y2, test_size=0.4, random_state=1)

    # Second split the 40% into validation and test sets
    X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1)

    #run training loop
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,   
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    model.compile(
        loss='mae',
        optimizer=optimizers.SGD(learning_rate=lr,momentum = 0.0, decay=decay, nesterov=nesterov))
    history = model.fit(np.array(X_train), np.array(y_train),
        validation_data=(np.array(X_val), np.array(y_val)),
            epochs=epochs, batch_size=batch_size,
            callbacks=[model_checkpoint_callback])

    #save model
    model.save(modelout)


def main(argv):
    modelin = ''
    modelout = ''
    imagepath = ''
    epochs = 1
    batch_size = 1
    lr = .01
    decay = 0.0
    nesterov = False
    checkpoint_filepath = "./"

    try:
        opts, args = getopt.getopt(argv,"hi:o:p:nd:l:b:e:c:",["modelin=","modelout=","imagepath=","nesterov","decay=","learningrate=","batchsize","epochs","checkpoint_filepath="])
    except getopt.GetoptError:
        print ('train.py -i <modelin> -o <modelout> -p <imagepath>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('train.py -i <modelin> -o <modelout> -p <imagepath>')
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
            lr = float(arg)
        elif opt in ("-b", "--batchsize"):
            batch_size = int(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)       
            
    print ('Input file is "', modelin)
    print ('Output file is "', modelout)
    print ('Image path is "', imagepath)

    if(modelin == '' or modelout == '' or imagepath == ''):
        print('Missing required parameter.')
        print ('train.py -i <modelin> -o <modelout> -p <imagepath>')
        sys.exit(2)


    print ('--------------------\n\n')

    train(modelin,modelout,imagepath,epochs,batch_size,lr,decay,nesterov,checkpoint_filepath)


if __name__ == "__main__":
    main(sys.argv[1:])