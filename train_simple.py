import os
import glob
import tensorflow as tf
import cv2
import keras
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, Input
from keras.layers import Convolution2D, MaxPooling2D, Reshape,Convolution3D, Conv2D, Bidirectional, add, MaxPool3D
from keras.layers import Input ,Dense, Dropout, Activation, LSTM, TimeDistributed, GRU
from keras.layers import Conv3D, Flatten, MaxPooling3D, average

from keras.models import model_from_json
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.layers import AveragePooling2D

from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D


#utility for training ==========================================================================================
#for ploting the accuracy and loss result
def plot_history(history, result_dir, name):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, '{}_accuracy.png'.format(name)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_loss.png'.format(name)))
    plt.close()

#to save model
def saveModel(model, model_name, outputDir):
    model_json = model.to_json()
    with open(os.path.join(outputDir, model_name+'.json'), 'w') as json_file:
        json_file.write(model_json)

#cnn existing model fine tune using keras, here you can see mobilenetv1, VGG19, VGG16 and resnet50 model mostly for classification
#this is the important part to study
def build_cnn(input_shape, nb_classes):
    mob_conv = keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #mob_conv = keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #mob_conv = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    #mob_conv = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))    
    x = mob_conv.output

    # Add new layers
    
    headModel = AveragePooling2D(pool_size=(7, 7))(x)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(1024, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dense(nb_classes, activation="softmax")(headModel)

    #trainable = 9
    #for layer in mob_conv.layers[:-trainable]:
        #layer.trainable = False
    #for layer in mob_conv.layers[-trainable:]:
        #layer.trainable = True

    #for layer in mob_conv.layers:
        #layer.trainable = False
    
    model = Model(inputs=mob_conv.input, outputs=headModel)

    return model

def blur(img):
    return (cv2.blur(img,(5,5)))

#setting the gpu fraction, if use CPU, no need this
gpu_memory_fraction = 0.5
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction 
sess = tf.Session(config=config)
keras.backend.set_session(sess)

#output directory for the model
outputDir = 'chkp_issa/'
model_name = 'gst_mod'

#load images using keras image generator ==============================================

# for data augmentation
data_aug = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    zoom_range=.2,
    #horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range = .2,
    #brightness_range=[0.2,0.8]
    #preprocessing_function= blur
    )

valid = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)
    #horizontal_flip=True)

#train data generator
train_generator = data_aug.flow_from_directory(
    directory=r"hand_images/train/", #folder of images to train
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32, #batch size is also important, if your computer is not have more memory, better to reduce the batch size number 
    class_mode="categorical",
    shuffle=True,
    seed=42
)

#validation data generator
valid_generator = valid.flow_from_directory(
    directory=r"hand_images/train/",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

#set the necessary parameters

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
classes = 4 #class number for gesture classification
input_shape = (224, 224, 3) #input shape of image to be train
epochs = 10 #how many epochs to train

#setup and prepare the model
model = build_cnn(input_shape,classes)

#optimizer for model training, its also important to change the learning rate tobe not too big or not too small
from keras.optimizers import Adam, SGD
learningRate = 0.001 
optimizer = Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

#compile the model for training
model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
#callback to check the loss or save the current weight
filepath=outputDir+"weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='auto', period=1)]

#save the model
model_json = model.to_json()
with open(os.path.join(outputDir, model_name+'.json'), 'w') as json_file:
    json_file.write(model_json)

#main model training code, here using fit generator that directly load the image from the folder
history = model.fit_generator(train_generator,steps_per_epoch=STEP_SIZE_TRAIN,epochs=epochs,validation_data=valid_generator,validation_steps=STEP_SIZE_VALID, callbacks=callbacks)

#ploting the result of accuracy and loss
plot_history(history, outputDir, 1)

#save the last model weights
model.save_weights(os.path.join(outputDir, model_name+'_last.hd5'))
model.save(os.path.join(outputDir, model_name+'_last.h5'))




