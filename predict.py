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
import math

def proc_image_dir(Images_Path, start, end):
    
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


            rawscore = int(os.path.basename(item).split("-")[0])

            if rawscore >= 1:
                if(j>=start and j<end+1):
                    # Read and resize image
                    full_size_image = io.imread(item)
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


    print("")
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


# ./input/
PATH = os.path.abspath(os.path.join('.', 'databaserelease2', 'flickr-2'))

# ./input/sample/images/
SOURCE_IMAGES = PATH#os.path.join(PATH, "sample", "images")
x2,y2,image_classes = proc_image_dir(SOURCE_IMAGES,1,100)




# First split the data in two sets, 60% for training, 40% for Val/Test)
#X_train, X_valtest, y_train, y_valtest = train_test_split(x2,y2, test_size=0.4, random_state=1)

# Second split the 40% into validation and test sets
#X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1)

#print(np.array(X_train).shape)
#print(np.array(X_val).shape)
#print(np.array(X_test).shape)


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


#physical_devices = tf.config.list_physical_devices('GPU') 
#print (physical_devices[1])
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

K.image_data_format()


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

run_filepath = './runs/run-20210609084750-3'
checkpoint_filepath = './runs/checkpointfile/'
history_path = run_filepath+".history"

print("Loading Model")
model = models.load_model(checkpoint_filepath)

print("Model Loaded")
a=np.array(x2).astype(float)
Y_pred = model.predict(a)

print(np.array(y2))
print(Y_pred)

y2_orig = []
for y in y2:
    y2_orig.append([y])
print(np.array(y2_orig))
y_comp = np.concatenate((np.array(y2_orig),Y_pred),axis=1)
print(y_comp)
y_diff = np.array(y2_orig) - Y_pred
print (y_diff)

'''
a=np.array(X_val).astype(float)
Y_pred = model.predict(a)

print(y_val)
print(Y_pred)
#model.load_weights(checkpoint_filepath+"checkpoint")
'''
#Y_pred = model.predict(a)

#print(Y_pred)

#pickle.dumps(history, open(history_path, 'w'))

