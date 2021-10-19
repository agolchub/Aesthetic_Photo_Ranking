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
import os
import tensorflow as tf
from tensorflow.python.platform.tf_logging import error

class WaitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            import time

            stream = os.popen("nvidia-smi -q -i 0 -d TEMPERATURE|grep 'GPU Current'|cut -d\" \" -f 30")
            temp = int(stream.read())
            print("The temperature of the GPU is ",temp)

            if(temp>75):
                print("Cooling down...")
                while(temp>55):
                    time.sleep(15)
                    stream = os.popen("nvidia-smi -q -i 0 -d TEMPERATURE|grep 'GPU Current'|cut -d\" \" -f 30")
                    temp = int(stream.read())

                print("Resuming -> ")

            return super().on_epoch_end(epoch, logs=logs)

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
        print(layer.name," re-initilized")
    except:
        print(layer.name, " could not be re-initilized", sys.exc_info())

def train(modelin,modelout,imagepath,epochs,batch_size,lr,decay,nesterov,checkpoint_filepath,train_path,val_path,transfer_learning,randomize_weights):
    #load model
    model = models.load_model(modelin)

    if transfer_learning:
        for layer in model.layers:
            if (isinstance(layer, tf.keras.layers.Conv2D)):
                print("conv layer",layer.name)
                layer.trainable = False

    if randomize_weights:
        
        for layer in model.layers:
            if(layer.trainable):
                init_layer(layer)

    model.build()

    print(model.summary())

    if(imagepath!=''):
        #load images
        x2,y2,images = proc_image_dir(imagepath)
        
        # First split the data in two sets, 60% for training, 40% for Val/Test)
        X_train, X_valtest, y_train, y_valtest = train_test_split(x2,y2, test_size=0.4, random_state=1)

        # Second split the 40% into validation and test sets
        X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1)
    else:
        X_train,y_train,image_list_train = proc_image_dir(train_path)
        X_val,y_val,image_list_val = proc_image_dir(val_path)

    #run training loop
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,   
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    wait_callback = WaitCallback()

    model.compile(
        loss='mae',
        optimizer=optimizers.SGD(learning_rate=lr,momentum = 0.0, decay=decay, nesterov=nesterov))
    history = model.fit(np.array(X_train), np.array(y_train),
        validation_data=(np.array(X_val), np.array(y_val)),
            epochs=epochs, batch_size=batch_size,
            callbacks=[model_checkpoint_callback,wait_callback])

    #save model
    model.save(modelout)

    print ([model.history.history["loss"],model.history.history["val_loss"]])

def main(argv):
    modelin = ''
    modelout = ''
    imagepath = ''
    epochs = 1
    batch_size = 1
    lr = .01
    decay = 0.0
    nesterov = False
    checkpoint_filepath = "./checkpoint/"
    train_path = ''
    val_path = ''
    transfer_learning = False
    randomize_weights = False

    try:
        opts, args = getopt.getopt(argv,"hi:o:p:nd:l:b:e:c:t:v:xr",["modelin=","modelout=","imagepath=","nesterov","decay=","learningrate=","batchsize","epochs","checkpoint_filepath=","train=","val=","test=","transfer_learning","randomize_weights"])
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
        elif opt in ("-t", "--train"):
            train_path = arg
        elif opt in ("-v", "--val"):
            val_path = arg  
        elif opt in ("-x", "--transfer_learning"):
            transfer_learning = True
        elif opt in ("-r", "--randomize_weights"):
            randomize_weights = True

    print ('Input file is "', modelin)
    print ('Output file is "', modelout)
    print ('Image path is "', imagepath)

    if(modelin == '' or modelout == '' or (imagepath == '' and (train_path == '' or val_path == ''))):
        print('Missing required parameter.')
        print ('train.py -i <modelin> -o <modelout> -p <imagepath>')
        sys.exit(2)


    print ('--------------------\n\n')

    train(modelin,modelout,imagepath,epochs,batch_size,lr,decay,nesterov,checkpoint_filepath,train_path,val_path,transfer_learning,randomize_weights)


if __name__ == "__main__":
    main(sys.argv[1:])


