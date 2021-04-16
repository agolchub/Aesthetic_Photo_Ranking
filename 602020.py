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

def proc_image_dir(Images_Path):
    
    image_classes = sorted([dirname for dirname in os.listdir(Images_Path)
                      if os.path.isdir(os.path.join(Images_Path, dirname)) and not dirname.startswith(".") and not dirname.startswith("mblur")])
    
    print(image_classes)
    
    x = [] # images as arrays
    y = [] # labels Infiltration or Not_infiltration
    WIDTH = 1650
    HEIGHT = 1100
  
    print("Adding Images: ",end="")
    i = 0
    for image_class in image_classes:
        print("Processing ", image_class)
        items = glob(os.path.join(Images_Path, image_class,"*"))
        j = 0
        for item in items:
            print("Reading "+item)
            if item.lower().endswith(".jpg") or item.lower().endswith(".bmp"):
                # Read and resize image
                full_size_image = io.imread(item)
                x.append(resize(full_size_image, (WIDTH,HEIGHT), anti_aliasing=True))
                #x.append(full_size_image)
                out = [0] * len(image_classes)
                out[i] = 1
                y.append(out)
                j+=1
                if(j>3): break
        i+=1

    print("")
    return x,y,image_classes

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
PATH = os.path.abspath(os.path.join('.', 'databaserelease2'))

# ./input/sample/images/
SOURCE_IMAGES = PATH#os.path.join(PATH, "sample", "images")
x2,y2,image_classes = proc_image_dir(SOURCE_IMAGES)




# First split the data in two sets, 60% for training, 40% for Val/Test)
X_train, X_valtest, y_train, y_valtest = train_test_split(x2,y2, test_size=0.4, random_state=1, stratify=y2)

# Second split the 40% into validation and test sets
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1, stratify=y_valtest)

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

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

K.image_data_format()

img_width, img_height = 1650, 1100
nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
epochs = 100
batch_size = 16

"""
l = layers.Conv2D(32, (132, 88), input_shape=(img_width, img_height, 3))(img)

l = layers.BatchNormalization()(l)
l = layers.Activation("softmax")(l)

l = layers.MaxPooling2D((2, 2))(l)

l = layers.Conv2D(64, (66, 44))(l)
l = layers.BatchNormalization()(l)
l = layers.Activation("softmax")(l)

l = layers.MaxPooling2D((2, 2))(l)

l = layers.Conv2D(128, (3, 2))(l)
l = layers.BatchNormalization()(l)
l = layers.Activation("softmax")(l)

l = layers.MaxPooling2D((2, 2))(l)

l = layers.Conv2D(128, (3, 2))(l)
l = layers.BatchNormalization()(l)
l = layers.Activation("softmax")(l)

l = layers.MaxPooling2D((2, 2))(l)

l = layers.Flatten()(l)
l = layers.Dropout(0.2)(l)
l = layers.Dense(64)(l)
l = layers.BatchNormalization()(l)
l = layers.Activation("relu")(l)
l = layers.Dropout(0.2)(l)

l = layers.Dense(len(image_classes))(l)
l = layers.BatchNormalization()(l)
preds = layers.Activation("sigmoid")(l)

labels = tf.placeholder(image_classes)

from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))
"""
 
model = models.Sequential()

model.add(layers.Conv2D(32, (264, 176), input_shape=(img_width, img_height, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("softmax"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (66, 44)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("softmax"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("softmax"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("softmax"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(len(image_classes)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("sigmoid"))

model.compile(
	loss='binary_crossentropy',
	optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
	metrics=['acc'])

model.summary() 

#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# Initialize all variables
#init_op = tf.global_variables_initializer()
#sess.run(init_op)

# Run training loop
#with sess.as_default():
#    for i in range(10):
#        batch = mnist_data.train.next_batch(50)
#        train_step.run(feed_dict={img: batch[0],
#                                  labels: batch[1]})
#print (np.array(X_train))

#print (tf.trainable_variables())

history = model.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_val), np.array(y_val)), epochs=50)