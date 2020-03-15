# -*- coding: utf-8 -*-
"""
About: A simple functioning Deep Learning model with trainining and validaion following 5.2 section of Deep Learning with Python
"""
import time
import cv2 as cv
import glob
import numpy as np
import os, shutil
# import tensorflow as tf ##to check if things are configured 
# print (tf.__version__)

dir_path="C:\\Users\\osama\\Desktop\\DATA 5000\\PROJECT\\code\\images\\opus_shuffled" #has all unsorted images
#create a folder "train": os.path.join gets path and the name of the new directory#os.mkdir(directory_name_path) makes it#You have to error check whether the folder already exist
#####Create a Test Dir#####
train_dir = os.path.join(dir_path, 'train') 
if os.path.exists(train_dir):
    pass #does nothing
else :
    os.mkdir(train_dir)
#####Create a Validation Dir#####
validation_dir = os.path.join(dir_path, 'validation') 
if os.path.exists(validation_dir):
    pass
else:
    os.mkdir(validation_dir)
####Create a Test Dir#####
test_dir = os.path.join(dir_path, 'test') 
if os.path.exists(test_dir):
    pass
else:
    os.mkdir(test_dir)
#GET IMAGES AND SHUFFLE
path = os.path.join(dir_path,'*jpg') #Adding *.jpg to the path
imgs = glob.glob(path);print('images in base_dir=',len(imgs))#read all images
np.random.shuffle(imgs) # to randomize images as per pg#132 Deep Learning F. Chollet. Allows to have similar rep. in train and test set


#DISPLAY AND SAVE IMAGES
count=1 #counter for running x times or naming images
for file in imgs: #each file in this case in an image
    if count<=200: #temp loop to just run 10 images
        img=cv.imread(file,-1) #create a variable that reads an image one by one (stores in the memory temporairly). 0 for BW. 1 for Color
        res_img = cv.resize(img, (150, 150))  # w,h
    
        #cv.imshow('img'+str(count),res_img)# shows img number
        #cv.imshow(str(file),res_img) #to display the image #. Helps in knowing whether the images have been shuffled
        if count<=160: #also used for sorting into folders
            isSaved = cv.imwrite('C:\\Users\\osama\\Desktop\\DATA 5000\\PROJECT\\code\\images\\opus_shuffled\\train\\train_shuf_'+str(count)+'.jpg', res_img)
            if isSaved:#checks of the image is saved
                pass#print('saved image opus_shuf_'+str(count)+'in the train folder')
        elif count>160 and count<=180:
            isSaved = cv.imwrite('C:\\Users\\osama\\Desktop\\DATA 5000\\PROJECT\\code\\images\\opus_shuffled\\validation\\val_shuf_'+str(count)+'.jpg', res_img)
            if isSaved:
                pass#print('saved image opus_shuf_'+str(count)+'in the val folder')
        else:
            isSaved = cv.imwrite('C:\\Users\\osama\\Desktop\\DATA 5000\\PROJECT\\code\\images\\opus_shuffled\\test\\test_shuf_'+str(count)+'.jpg', res_img)
            if isSaved:
                pass#print('saved image opus_shuf_'+str(count)+'in the test folder')      
        
        cv.waitKey(1) #in ms
        cv.destroyAllWindows()
        #print (count)
        count = count+1

#LIBRARIES, SETTING UP TENSORFLOW AND KERAS
import tensorflow as tf; print('tf_version:',tf.__version__) #tf: 2.1
from tensorflow import keras; print('keras version:',keras.__version__)
from keras import layers; from keras import models; from keras import optimizers

#SETTING UP THE MODEL FOR TRAINING//LAYERS+ACTIVATION FUNCTIONS

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
#COMPIING THE MODEL//optimizer
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


#DATA PREPROCESSING WITH (KERAS)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)#RESCALE IMAGES
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150), #RESIZE
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=5,
        class_mode='binary')


#COMPILING OR FITTING THE MODEL USING THE BATCH GENERATOR
history = model.fit_generator(
      train_generator,
      steps_per_epoch=50,
      epochs=3,
      validation_data=validation_generator,
      validation_steps=5)
#SAVE THE MODEL
model.save('opus_v4.h5')

#CHECK THE ACCURACY
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()