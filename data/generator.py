import csv
from skimage import io
import os
import gc
from skimage.transform import resize
import numpy as np
import skimage
import tensorflow as tf
from threading import Thread, Lock
imageProcessingLock = Lock()
import random

def read_one_image(Images_Path, image, score, HEIGHT, WIDTH, categorical, categories, y, x):
    resizedImage = None
    out = None
    try:
        imagePath = Images_Path + image
        full_size_image = io.imread(imagePath)
        resizedImage = resize(full_size_image, (WIDTH, HEIGHT), anti_aliasing=True)
        if resizedImage.shape[2] == 3:
            if (categorical):
                out = [0] * categories
                out[int(score)] = 1
            else:
                out = float(score)  # ((rawscore - 1.0)/9.0)
        del full_size_image

        imageProcessingLock.acquire()
        try:
            y.append(out)
            x.append(resizedImage.astype(np.float16))
        except Exception as e:
            print("Error ---- ")
            print(e)
        finally:
            imageProcessingLock.release()
    except Exception as e:
        print("Error ---- ")
        print(e)
    #gc.collect()
    return resizedImage, out

class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self,
                 batch_size, path,
                 input_size=(1024, 680, 3),
                 shuffle=False, resnet=False, outColumn = 2, categorical=False, categories=1, epoch_length=0):
        self.categorical = categorical
        self.categories = categories
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.path = path
        self.items = []
        self.images_path = os.path.dirname(path) + '/JPEG/'
        with open(path, mode='r') as cat:
            csvFile = csv.reader(cat)
            for line in csvFile:
                self.items.append([line[0],line[outColumn]])
        if(self.shuffle):
            print("shuffling")
            random.shuffle(self.items)

        self.n = len(self.items)

        if(epoch_length>0 and epoch_length<self.n):
            self.n = epoch_length

        self.resnet = resnet

        threads=[]
        x = []  # images as arrays
        y = []  # labels Infiltration or Not_infiltration
        WIDTH = 1024
        HEIGHT = 680
        for item in self.items:
            t = Thread(target=read_one_image,args=(self.images_path, item[0], item[1], HEIGHT, WIDTH, self.categorical, self.categories, y, x))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        gc.collect()
        if self.resnet:
            self.x = tf.keras.applications.resnet.preprocess_input(x)

        

        self.x=x
        self.y=y


    def on_epoch_end(self):
        # if(self.shuffle):
        #     print("shuffling")
        #     random.shuffle(self.items)
        temp = list(zip(self.x,self.y))
        random.shuffle(temp)
        self.x, self.y = zip(*temp)
        self.x, self.y = list(self.x), list(self.y)

    def __getitem__(self, index):
        # # import random

        # x = []  # images as arrays
        # y = []  # labels Infiltration or Not_infiltration
        # WIDTH = 1024
        # HEIGHT = 680
        # if self.resnet:
        #     WIDTH = HEIGHT = 224
        # j = 0
        # images = []

        # threads = []
        # for i in range(index * self.batch_size, (index + 1) * self.batch_size):
        #     if (i >= len(self.items)):
        #         break
        #     item = self.items[i]
        #     t = Thread(target=read_one_image,args=(self.images_path, item[0], item[1], HEIGHT, WIDTH, self.categorical, self.categories, y, x))
        #     t.start()
        #     threads.append(t)
        # for t in threads:
        #     t.join()
        # gc.collect()
        # if self.resnet:
        #     x = tf.keras.applications.resnet.preprocess_input(x)
        from datetime import datetime
        x = []
        y = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            if (i >= len(self.x)):
                break
            x.append(self.x[i])
            y.append(self.y[i])
        return tf.convert_to_tensor(x,dtype=tf.float16), tf.convert_to_tensor(y,dtype=tf.float16)

    def __len__(self):
        return self.n // self.batch_size
    


