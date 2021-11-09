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

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self,
                 batch_size,path,
                 input_size=(1024, 680, 3),
                 shuffle=True,resnet=False):
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.path = path
        self.items = glob(os.path.join(self.path,"*.jpg"))
        self.n = len(self.items)
        self.resnet = resnet
    
    def on_epoch_end(self):
        pass
    
    def __getitem__(self, index):
        #import random

        x = [] # images as arrays
        y = [] # labels Infiltration or Not_infiltration
        WIDTH = 1024
        HEIGHT = 680
        if self.resnet:
            WIDTH=HEIGHT=224
        j = 0
        images = []
        rawscore = 0.0

        for i in range(index*self.batch_size, (index+1)*self.batch_size):
            if(i>=len(self.items)):
                break
            item = self.items[i]
            #print("Reading " + item)
            # Read and resize image
            full_size_image = io.imread(item)
            rawscore = float(os.path.basename(item).split("-")[0])
            out = rawscore
            y.append(out)
            images.append(item)
            resizedImage = resize(full_size_image, (WIDTH,HEIGHT), anti_aliasing=True) 
            if(len(resizedImage.shape) < 3):
                resizedImage = skimage.color.gray2rgb(resizedImage)

            x.append(resizedImage)
        x = np.array(x)
        y = np.array(y)
        if self.resnet:
            x = tf.keras.applications.resnet.preprocess_input(x)

        return x,y
    
    def __len__(self):
        return self.n // self.batch_size

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

def proc_image_dir(Images_Path,categorical=False):
    import random
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
    random.shuffle(items)
    #items = items[:200]
    for item in items:
        print("Reading "+item)
        if item.lower().endswith(".jpg") or item.lower().endswith(".bmp"):
            # Read and resize image
            full_size_image = io.imread(item)

            rawscore = float(os.path.basename(item).split("-")[0])

            if rawscore >= 0:
                #x.append(full_size_image)
                if(categorical):
                    out = [0]*5
                    out[int(rawscore)-1] = 1
                else:
                    out = ((rawscore - 1.0)/9.0)
                print(out)
                y.append(out)
                images.append(item)
                resizedImage = resize(full_size_image, (WIDTH,HEIGHT), anti_aliasing=True) 
                if(len(resizedImage.shape) < 3):
                    resizedImage = skimage.color.gray2rgb(resizedImage)
                resizedImage = tf.keras.applications.resnet.preprocess_input(resizedImage)
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

def new_conv2d(input,n,size=(2,2),strides=(2,2),activation="relu",kernel_initializer="glorot_uniform",batch_normalization=True,dropout_rate=0.2):
    conv2d = layers.Conv2D(n, size, strides=strides, kernel_initializer=kernel_initializer)(input)
    if(batch_normalization):
        batchNormalization = layers.BatchNormalization()(conv2d)
    else:
        batchNormalization = conv2d
    activationLayer = layers.Activation(activation)(batchNormalization)
    dropout = activationLayer#layers.Dropout(dropout_rate)(activationLayer)
    return dropout

def new_dense(input,n,activation="relu",kernel_initializer="glorot_uniform",dropout_rate=0.2):
    dense = layers.Dense(n, activation=activation,kernel_initializer=kernel_initializer)(input)
    dropout = layers.Dropout(dropout_rate)(dense)
    return dropout

def new_res_block(input,n,m,size,strides):
    conv2d = layers.Conv2D(n, size, strides=strides)(input)
    batchNormalization = layers.BatchNormalization()(conv2d)
    activation = layers.Activation("relu")(batchNormalization)
    shortcut = activation
    conv2d = layers.Conv2D(n*m, (2,2), strides=(1,1))(activation)
    batchNormalization = layers.BatchNormalization()(conv2d)
    activation = layers.Activation("relu")(batchNormalization)
    conv2d = layers.Conv2D(n*m*m, (2,2), strides=(1,1))(activation)
    batchNormalization = layers.BatchNormalization()(conv2d)

    shortcut = layers.Conv2D(n*m*m, (3,3), strides=(1,1))(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    add = layers.Add()([batchNormalization,shortcut])
    activation = layers.Activation("relu")(add)
    return activation


def train(modelin,modelout,imagepath,epochs,batch_size,lr,decay,nesterov,checkpoint_filepath,train_path,val_path,transfer_learning,randomize_weights,use_resnet,special_model,build_only,special_model2,batched_reader,simple_model,momentum,loss_function):
    #load model
    if(use_resnet):
        resnetmodel = tf.keras.applications.resnet50.ResNet50(input_shape=(224,224,3),include_top=False)
        for layer in resnetmodel.layers:
            layer.trainable=False
        f1 = layers.Flatten()(resnetmodel.output)

        l1 = Lambda(lambda x: x[:,0:10000])(f1)
        l2 = Lambda(lambda x: x[:,10000:20000])(f1)
        l3 = Lambda(lambda x: x[:,20000:30000])(f1)
        l4 = Lambda(lambda x: x[:,30000:40000])(f1)
        l5 = Lambda(lambda x: x[:,40000:50000])(f1)
        l6 = Lambda(lambda x: x[:,50000:60000])(f1)
        l7 = Lambda(lambda x: x[:,60000:70000])(f1)
        l8 = Lambda(lambda x: x[:,70000:80000])(f1)
        l9 = Lambda(lambda x: x[:,80000:90000])(f1)
        l10 = Lambda(lambda x: x[:,90000:])(f1)

        d1_1 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(l1)
        d1_2 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(l2)
        d1_3 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(l3)
        d1_4 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(l4)
        d1_5 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(l5)
        d1_6 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(l6)
        d1_7 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(l7)
        d1_8 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(l8)
        d1_9 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(l9)
        d1_10 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(l10)

        #d1 = layers.Concatenate()([d1_1,d1_2,d1_3,d1_4,d1_5,d1_6,d1_7,d1_8,d1_9,d1_10])
        d1= layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(f1)
        do1= Dropout(0.5)(d1)
        d2 = layers.Dense(128, activation="relu",kernel_initializer="he_uniform")(do1)
        do2= layers.Dropout(0.2)(d2)
        d2 = layers.Dense(128, activation="relu",kernel_initializer="he_uniform")(do2)
        do2= layers.Dropout(0.2)(d2)
        d2 = layers.Dense(128, activation="relu",kernel_initializer="he_uniform")(do2)
        do2= layers.Dropout(0.2)(d2)
        d3 = Dense(1, kernel_initializer="he_uniform", activation="relu")(do2)
        
        model = models.Model(inputs=resnetmodel.input,outputs=d3)
    elif(simple_model):
        input = layers.Input((1024,680,3))
        do0 = new_conv2d(input,80,(36,12),(15,10))

        f1   = layers.Flatten()(do0)

        do1 = new_dense(f1, 64,dropout_rate=0.5)
        do2   = new_dense(do1, 64)
        d4    = Dense(1, kernel_initializer="he_uniform", activation="linear")(do2)
        model = models.Model(inputs=input,outputs=d4)
    elif(special_model):
        input = layers.Input((1024,680,3))
        c1    = layers.Conv2D(32, (132, 88),input_shape=(1024, 680, 3), strides=(3,2), activation="relu", kernel_initializer="he_uniform")(input)
        b1    = layers.BatchNormalization()(c1)
        do1   = layers.Dropout(0.2)(b1)

        c2    = layers.Conv2D(64, (66, 66), strides=(2,2), activation="relu",kernel_initializer="he_uniform")(do1)
        b2    = layers.BatchNormalization()(c2)
        do2   = layers.Dropout(0.5)(b2)

        c3    = layers.Conv2D(128, (16, 16), strides=(2,2), activation="relu",kernel_initializer="he_uniform")(do2)
        b3    = layers.BatchNormalization()(c3)
        do3   = layers.Dropout(0.5)(b3)

        c4    = layers.Conv2D(256, (7, 7), strides=(2,2), activation="relu",kernel_initializer="he_uniform")(do3)
        b4    = layers.BatchNormalization()(c4)
        do4   = layers.Dropout(0.5)(b4)

        c5    = layers.Conv2D(512, (3, 3), strides=(1,1), activation="relu",kernel_initializer="he_uniform")(do4)
        b5    = layers.BatchNormalization()(c5)
        do5   = layers.Dropout(0.5)(b5)

        c6    = layers.Conv2D(256, (2, 2), strides=(1,1), activation="relu",kernel_initializer="he_uniform")(do5)
        b6    = layers.BatchNormalization()(c6)
        do6   = layers.Dropout(0.2)(b6)

        c7    = layers.Conv2D(128, (2, 2), strides=(1,1), activation="relu",kernel_initializer="he_uniform")(do6)
        b7    = layers.BatchNormalization()(c7)
        do7   = layers.Dropout(0.2)(b7)

        c8    = layers.Conv2D(64, (2, 2), strides=(1,1), activation="relu",kernel_initializer="he_uniform")(do7)
        b8    = layers.BatchNormalization()(c8)
        do8   = layers.Dropout(0.2)(b8)

        c9    = layers.Conv2D(64, (2, 2), strides=(1,1), activation="relu",kernel_initializer="he_uniform")(do8)
        b9    = layers.BatchNormalization()(c9)
        do9   = layers.Dropout(0.2)(b9)

        f1_1  = layers.Flatten()(do1)
        f1_2  = layers.Flatten()(do2)
        f1_3  = layers.Flatten()(do3)
        f1_4  = layers.Flatten()(do4)
        f1_5  = layers.Flatten()(do5)
        
        f1_6  = layers.Flatten()(do6)
        f1_7  = layers.Flatten()(do7)
        f1_8  = layers.Flatten()(do8)

        f1_9  = layers.Flatten()(do9)

        d0_1 = layers.Dense(5, activation="relu",kernel_initializer="he_uniform")(f1_1)
        d0_2 = layers.Dense(5, activation="relu",kernel_initializer="he_uniform")(f1_2)
        d0_3 = layers.Dense(5, activation="relu",kernel_initializer="he_uniform")(f1_3)
        d0_4 = layers.Dense(5, activation="relu",kernel_initializer="he_uniform")(f1_4)
        d0_5 = layers.Dense(5, activation="relu",kernel_initializer="he_uniform")(f1_5)
        d0_6 = layers.Dense(5, activation="relu",kernel_initializer="he_uniform")(f1_6)
        d0_7 = layers.Dense(5, activation="relu",kernel_initializer="he_uniform")(f1_7)
        d0_8 = layers.Dense(5, activation="relu",kernel_initializer="he_uniform")(f1_8)
        d0_9 = layers.Dense(128, activation="relu",kernel_initializer="he_uniform")(f1_9)

        f1   = layers.Concatenate()([d0_1,d0_2,d0_3,d0_4,d0_5,d0_6,d0_7,d0_8,d0_9])

        d1 = layers.Dense(256, activation="relu",kernel_initializer="he_uniform")(f1_9)
        do1   = Dropout(0.5)(d1)
        d2    = layers.Dense(128, activation="relu",kernel_initializer="he_uniform")(do1)
        do2   = layers.Dropout(0.2)(d2)
        d3    = Dense(10, kernel_initializer="he_uniform", activation="softmax")(do2)
        model = models.Model(inputs=input,outputs=d3)

    elif(special_model2):
        input = layers.Input((1024,680,3))
        resblock = new_res_block(input,96,2,(35,35),(6,4))
        resblock = new_res_block(resblock,384,0.5,(9,9),(2,2))
        resblock = new_res_block(resblock,384,2,(5,5),(2,2))
        resblock = new_res_block(resblock,384,0.5,(3,3),(2,2))
        resblock = new_res_block(resblock,160,1,(3,3),(2,2))

        resblock2 = new_res_block(input,96,2,(9,9),(6,4))
        resblock2 = new_res_block(resblock2,384,0.5,(7,7),(2,2))
        resblock2 = new_res_block(resblock2,384,2,(5,5),(2,2))
        resblock2 = new_res_block(resblock2,384,0.5,(3,3),(2,2))
        resblock2 = new_res_block(resblock2,160,1,(3,3),(2,2))
        '''
        do0 = new_conv2d(input,96,(35,35),(4,4))
        do0_1 = layers.MaxPooling2D(pool_size=(2,2))(do0)
        do0_1   = new_conv2d(do0,256,(9,9),(2,2))
        do0_1   = new_conv2d(do0_1,384,(3,3),(2,2))
        do0_1   = new_conv2d(do0_1,384,(2,2),(2,2))
        do0_1 = layers.MaxPooling2D(pool_size=(2,2))(do0_1)

        ##
        do1 = new_conv2d(input,96,(11, 11), strides=(3,2))
        do1 = layers.MaxPooling2D(pool_size=(2,2))(do1)
        do1_1 = new_conv2d(do1,80, (9, 9), strides=(2,2))
        do1_1 = new_conv2d(do1_1,160, (9, 9), strides=(2,2))

        ##
        do2   = new_conv2d(do1,160, (5, 5), strides=(2,2))
        do2_1   = new_conv2d(do2,160, (5, 5), strides=(2,2))
        do2_1   = new_conv2d(do2_1,320, (5, 5), strides=(2,2))

        ##
        do3   = new_conv2d(do2,160, (3, 3), strides=(2,2))
        do3_1   = new_conv2d(do3,160, (7, 7), strides=(2,2))
        do3_1   = new_conv2d(do3_1,320, (7, 7), strides=(2,2))

        ##
        do4   = new_conv2d(do3,160, (3, 3), strides=(2,2))
        do4_1   = new_conv2d(do4,320, (3, 3), strides=(2,2))
        do4_1   = new_conv2d(do4_1,320, (3, 3), strides=(2,2))

        ##
        do5   = new_conv2d(do4,320, (2, 2), strides=(2,2))
        do5_1   = new_conv2d(do5,320, (2, 2), strides=(2,2))
        do5_1   = new_conv2d(do5_1,320, (2, 2), strides=(2,2))

        ##
        do6   = new_conv2d(do5,320, (2, 2), strides=(2,2))
        do6_1   = new_conv2d(do6,320, (2, 2), strides=(2,2))

        ##
        f0_1  = layers.Flatten()(do0_1)
        f1_1  = layers.Flatten()(do1_1)
        f1_2  = layers.Flatten()(do2_1)
        f1_3  = layers.Flatten()(do3_1)
        f1_4  = layers.Flatten()(do4_1)
        f1_5  = layers.Flatten()(do5_1)   
        f1_6  = layers.Flatten()(do6_1)

        d0_0 = new_dense(f0_1, 256)
        d0_1 = new_dense(f1_1, 25)
        d0_2 = new_dense(f1_2, 25)
        d0_3 = new_dense(f1_3, 25)
        d0_4 = new_dense(f1_4, 25)
        d0_5 = new_dense(f1_5, 25)
        d0_6 = new_dense(f1_6, 256)

        '''
        f1   = layers.Concatenate()([layers.Flatten()(resblock),layers.Flatten()(resblock2)]) #d0_1,d0_2,d0_3,d0_4,d0_5,
        
        #f1 = layers.Flatten()(f1)
        do1 = new_dense(f1, 2048,dropout_rate=0.2, activation="relu")
        do2   = new_dense(do1, 1024, activation="relu")
        d3    = new_dense(do2, 512, activation="relu")
        d3    = new_dense(d3, 256, activation="relu")
        d3    = new_dense(d3, 128, activation="relu")
        d3    = new_dense(d3, 64, activation="relu")
        d4    = Dense(1, activation="sigmoid")(d3)
        model = models.Model(inputs=input,outputs=d4)
    else:
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

    # Define the Keras TensorBoard callback.
    logdir=modelout+".logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    #model.build()
    model.compile(
        loss='mse',
        optimizer=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))

    print(model.summary())
    if(build_only):
        img_file = modelout+'-model_arch.png'
        tf.keras.utils.plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)
        model.save(modelout)
        exit(0)

    if(imagepath!=''):
        #load images
        x2,y2,images = proc_image_dir(imagepath)
        
        # First split the data in two sets, 60% for training, 40% for Val/Test)
        X_train, X_valtest, y_train, y_valtest = train_test_split(x2,y2, test_size=0.4, random_state=1)

        # Second split the 40% into validation and test sets
        X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1)
    elif(batched_reader):
        training_generator = CustomDataGen(batch_size,train_path,resnet=use_resnet)
        validation_generator = CustomDataGen(batch_size,val_path,resnet=use_resnet)
    else:
        X_train,y_train,image_list_train = proc_image_dir(train_path)
        X_val,y_val,image_list_val = proc_image_dir(val_path)

    #run training loop
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,   
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    wait_callback = WaitCallback()

    model.compile(
        loss=loss_function,
        optimizer=optimizers.SGD(learning_rate=lr,momentum = momentum, decay=decay, nesterov=nesterov),
        metrics=['accuracy'])
    #model.build()
    #history = model.fit(np.array(X_train), np.array(y_train),
    #    validation_data=(np.array(X_val), np.array(y_val)),
    #        epochs=epochs, batch_size=batch_size,
    #        callbacks=[model_checkpoint_callback,wait_callback])
    if(batched_reader):
        history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=epochs,callbacks=[model_checkpoint_callback,wait_callback,tensorboard_callback])
    else:
        history = model.fit(np.array(X_train), np.array(y_train),
        validation_data=(np.array(X_val), np.array(y_val)),
            epochs=epochs, batch_size=batch_size,
            callbacks=[model_checkpoint_callback,wait_callback,tensorboard_callback])

    #save model
    model.save(modelout)

    print ([model.history.history["loss"],model.history.history["val_loss"]])

def test(modelin,imagepath):
    model = models.load_model(modelin)
    #model.load_weights(modelin+".checkpoint/")
    print("Model Loaded")
    print(model.summary())
    x,y,images = proc_image_dir(imagepath)
    a=np.array(x).astype(float)
    Y_pred = model.predict(a)
    print(np.array(y))
    print(Y_pred)

    np.set_printoptions(threshold=sys.maxsize)

    print(np.argmax(np.array(y),axis=1))
    print(np.argmax(Y_pred,axis=1))

    loss, acc = model.evaluate(a, np.array(y), verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

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
    use_resnet = False
    special_model = False
    build_only = False
    testmode = False
    special_model2 = False
    batched_reader = False
    simple_model = False
    momentum = 0.0
    loss_function = 'mse'

    try:
        opts, args = getopt.getopt(argv,"hi:o:p:nd:l:b:e:c:t:v:xrm:f:",["loss=","momentum=","special_model2","simple_model","test","build_only","modelin=","resnet50","special_model","modelout=","imagepath=","nesterov","decay=","learningrate=","batchsize","epochs","checkpoint_filepath=","train=","val=","test=","transfer_learning","randomize_weights","batched_reader"])
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
        elif opt in ("-e","--momentum"):
            momentum = arg
        elif opt in ("-f","--loss"):
            loss_function = arg

    checkpoint_filepath = modelout+".checkpoint/"

    print ('Input file is "', modelin)
    print ('Output file is "', modelout)
    print ('Image path is "', imagepath)

    if(not testmode and ((modelin == '' and not use_resnet and not special_model and not special_model2 and not simple_model) or modelout == '' or (imagepath == '' and (train_path == '' or val_path == '')))):
        print('Missing required parameter.')
        print ('train.py -i <modelin> -o <modelout> -p <imagepath>')
        sys.exit(2)

    if(testmode and (modelin == '' or imagepath == '')):
        print('Missing required parameter.')
        print ('train.py --test -i <modelin> -p <imagepath>')
        sys.exit(2)

    print ('--------------------\n\n')

    if(testmode):
        test(modelin,imagepath)
    else:
        train(modelin,modelout,imagepath,epochs,batch_size,lr,decay,nesterov,checkpoint_filepath,train_path,val_path,transfer_learning,randomize_weights,use_resnet,special_model,build_only,special_model2,batched_reader,simple_model,momentum,loss_function)

if __name__ == "__main__":
    main(sys.argv[1:])
