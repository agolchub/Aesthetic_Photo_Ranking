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
x2,y2,image_classes = proc_image_dir(SOURCE_IMAGES)




# First split the data in two sets, 60% for training, 40% for Val/Test)
X_train, X_valtest, y_train, y_valtest = train_test_split(x2,y2, test_size=0.4, random_state=1)

# Second split the 40% into validation and test sets
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1)

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

img_width, img_height = 1024, 680
nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
epochs = 50
batch_size = 10

initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)

model = models.Sequential()
'''
model.add(layers.Conv2D(32, (128, 85), input_shape=(img_width, img_height, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("softmax"))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (66, 44)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("softmax"))

model.add(layers.MaxPooling2D((2, 2)))
'''
model.add(layers.Conv2D(32, (132, 88),input_shape=(img_width, img_height, 3), strides=(4,4), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.2))

#model.add(layers.MaxPooling2D((4, 4)))

model.add(layers.Conv2D(64, (66, 44), strides=(4,4), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Conv2D(128, (33, 22), strides=(4,4), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
'''
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("softmax"))
model.add(layers.Dropout(0.5))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Activation("softmax"))
model.add(layers.Dropout(0.5))
'''

model.add(layers.Flatten())

model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.2))
#model.add(layers.BatchNormalization())



'''
model.add(layers.Dense(128))
#model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dense(128))
#model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dense(64))
#model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
#model.add(layers.Dropout(0.2))
'''
model.add(Dense(1, input_dim=1, kernel_initializer='glorot_uniform', activation='linear'))

model.compile(
	loss='mse',
	optimizer=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

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

#tf.keras.utils.plot_model(
#    model,
#    to_file="model.png",
#    show_shapes=False,
#    show_dtype=False,
#    show_layer_names=True,
#    rankdir="TB",
#    expand_nested=False,
#    dpi=96,
#)


run_filepath = './runs.log/run-'+datetime.now().strftime("%Y%m%d%H%M%S")
checkpoint_filepath = 'runs.log/checkpointfile/'
history_path = run_filepath+".history"

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,   
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=False)

model.compile(
	loss='mae',
	optimizer=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
history = model.fit(np.array(X_train), np.array(y_train),
    validation_data=(np.array(X_val), np.array(y_val)),
        epochs=30, batch_size=batch_size,
        callbacks=[model_checkpoint_callback])
model.save(run_filepath+'-1')

model.compile(
	loss='mae',
	optimizer=optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
history = model.fit(np.array(X_train), np.array(y_train),
    validation_data=(np.array(X_val), np.array(y_val)),
        epochs=60, batch_size=batch_size,
        callbacks=[model_checkpoint_callback])
model.save(run_filepath+'-2')

model.compile(
	loss='mae',
	optimizer=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
history = model.fit(np.array(X_train), np.array(y_train),
    validation_data=(np.array(X_val), np.array(y_val)),
        epochs=epochs, batch_size=batch_size,
        callbacks=[model_checkpoint_callback])
model.save(run_filepath+'-3')


model.save(run_filepath)

a=np.array(X_test).astype(float)
Y_pred = model.predict(a)

print(y_test)
print(Y_pred)

model.load_weights(checkpoint_filepath+"checkpoint")

Y_pred = model.predict(a)

print(Y_pred)


#pickle.dumps(history, open(history_path, 'w'))

