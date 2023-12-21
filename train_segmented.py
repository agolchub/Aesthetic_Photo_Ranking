#!/usr/bin/env python

import skimage
import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize
import sys, getopt
from tensorflow.keras import initializers, models, optimizers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import os
import tensorflow as tf
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.python.lib.io.file_io import file_crc32
from tensorflow.python.platform.tf_logging import error

from datetime import datetime
import matplotlib.pylab as plt
import itertools

import models as myModels
from models.primitives import init_layer

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

import threading
imageProcessingLock = threading.Lock()
def proc_image_dir(Images_Path, scores="", categorical=False, WIDTH=1024, HEIGHT=680, scoreColumn=5, categories=5, sample=10000):
    import random
    import csv
    import os
    import psutil
    import gc
    from threading import Thread
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
    threads = []
    count = 0
    if scores != "":
        with open(scores, mode='r') as cat:
            csvFile = csv.reader(cat)
            for line in csvFile:
                count=count+1
                if(not (count%sample == 0)):
                    t = Thread(target=read_one_image,args=(HEIGHT, Images_Path, WIDTH, categorical, line, scoreColumn, categories, y, x, images))
                    t.start()
                    threads.append(t)
            for t in threads:
                t.join()
            gc.collect()
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


def read_one_image(HEIGHT, Images_Path, WIDTH, categorical, line, scoreColumn, categories, y, x, images):
    import gc
    try:
        imagePath = Images_Path + line[0]
        full_size_image = io.imread(imagePath)
        resizedImage = resize(full_size_image, (WIDTH, HEIGHT), anti_aliasing=True)
        if resizedImage.shape[2] == 3:
            if (categorical):
                out = [0] * categories
                out[int(line[scoreColumn])] = 1
            else:
                out = float(line[scoreColumn])  # ((rawscore - 1.0)/9.0)

            print(line[scoreColumn] + " - " + imagePath)
        del full_size_image

        imageProcessingLock.acquire()
        try:
            y.append(out)
            x.append(resizedImage)
            images.append(Images_Path + line[0])
        except Exception as e:
            print("Error ---- ")
            print(e)
            print(line)
        finally:
            imageProcessingLock.release()
    except Exception as e:
        print("Error ---- ")
        print(e)
        print(line)
    #gc.collect()
    return resizedImage, out

def train(modelin, modelout, imagepath, epochs, batch_size, lr, decay, nesterov, checkpoint_filepath, train_path,
          val_path, transfer_learning, randomize_weights, use_resnet, special_model, build_only, special_model2,
          batched_reader, simple_model, momentum, loss_function, catalog, WIDTH, HEIGHT, outColumn,
          unlock_segment_weights, model_design, reload, patience):
    categorical = False
    # load model

    if (simple_model):
        model = myModels.simple_model(WIDTH, HEIGHT)
        
    elif (special_model):
        model = myModels.special_model(WIDTH, HEIGHT, unlock_segment_weights)

    elif (special_model2):
        model = myModels.special_model2(WIDTH, HEIGHT, unlock_segment_weights)

    elif model_design == 3:
        model = myModels.model_3(WIDTH, HEIGHT)

    elif model_design == 4:
        model = myModels.model_4(WIDTH, HEIGHT)

    elif model_design == 5:
        model = myModels.model_5(WIDTH, HEIGHT)

    elif model_design == 6:
        model = myModels.model_6(WIDTH, HEIGHT)

    elif model_design == 7:
        model = myModels.model_7(WIDTH, HEIGHT)

    elif model_design == 8:
        model = myModels.model_8(WIDTH, HEIGHT)

    elif model_design == 9:
        model = myModels.model_9(WIDTH, HEIGHT)

    elif model_design == 10:
        model = myModels.model_10(WIDTH, HEIGHT)

    elif model_design == 11:
        model = myModels.model_11(WIDTH, HEIGHT)

    elif model_design == 12:
        model = myModels.model_12(WIDTH, HEIGHT)

    elif model_design == 13:
        model = myModels.model_13(WIDTH, HEIGHT)

    elif model_design == 14:
        model = myModels.model_14(WIDTH, HEIGHT)

    elif model_design == 15:
        model = myModels.model_15(WIDTH, HEIGHT)

    elif model_design == 16:
        model = myModels.model_16(WIDTH, HEIGHT)

    elif model_design == 17:
        model = myModels.model_17(WIDTH, HEIGHT)

    elif model_design == 18:
        model = myModels.model_18(WIDTH, HEIGHT)

    elif model_design == 19:
        model = myModels.model_19(WIDTH, HEIGHT)

    elif model_design == 20:
        model = myModels.model_20(WIDTH, HEIGHT)

    elif model_design == 21:
        model = myModels.model_21(WIDTH, HEIGHT)

    elif model_design == 22:
        model = myModels.model_22(WIDTH, HEIGHT)

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
        optimizer=optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, weight_decay=0.0))

    print(model.summary())
    if (build_only):
        img_file = f'model_archs/{modelout.split("/")[-1]}-model_arch.png'
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
                                                            scoreColumn=outColumn, categorical=categorical, categories=model.output_shape[1], sample=2)
        X_val, y_val, image_list_val = proc_image_dir(os.path.dirname(os.path.abspath(val_path)) + '/JPEG/', val_path,
                                                      WIDTH=WIDTH, HEIGHT=HEIGHT, scoreColumn=outColumn, categorical=categorical, categories=model.output_shape[1], sample=2)

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
                                  HEIGHT=HEIGHT, scoreColumn=outColumn, categorical=categorical, categories=model.output_shape[1])
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
