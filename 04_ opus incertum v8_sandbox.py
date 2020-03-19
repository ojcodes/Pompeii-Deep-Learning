# SETTING UP TENSORFLOW AND KERAS
import sys;print('python version:',sys.version)
import tensorflow as tf; print('tf_version:',tf.__version__) #tf: 2.1
from tensorflow import keras; print('keras version:',keras.__version__)
from tensorflow.python.platform import build_info as tf_build_info
print('CUDA Version:',tf_build_info.cuda_version_number)
print('CUDNN Version:',tf_build_info.cudnn_version_number)


import os, shutil
original_dataset_dir="C:\\Users\\osama\\Documents\\opus_shuffled" #has all unsorted images

####Create train, validation and test folders and sub-folders. Keras needs sub-folders:https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# Create a Test Dir##### #create a folder "train": os.path.join gets path and the name of the new directory#os.mkdir(directory_name_path) makes it#You have to error check whether the folder already exist
train_dir = os.path.join(original_dataset_dir, 'train') 
if os.path.exists(train_dir):
    pass #does nothing
else :
    os.mkdir(train_dir)

# train_sub_dir = os.path.join(train_dir, 'train_sub') 
# if os.path.exists(train_sub_dir):
#     pass #does nothing
# else :
#     os.mkdir(train_sub_dir)
    
# #####Create a Validation Dir#####
validation_dir = os.path.join(original_dataset_dir, 'validation') 
if os.path.exists(validation_dir):
    pass #does nothing
else :
    os.mkdir(validation_dir)

# validation_sub_dir = os.path.join(validation_dir, 'validation_sub') 
# if os.path.exists(validation_sub_dir):
#     pass #does nothing
# else :
#     os.mkdir(validation_sub_dir)

# ####Create a Test Dir#####
test_dir = os.path.join(original_dataset_dir, 'test') 
if os.path.exists(test_dir):
    pass #does nothing
else :
    os.mkdir(test_dir)

# test_sub_dir = os.path.join(test_dir, 'test_sub') 
# if os.path.exists(test_sub_dir):
#     pass #does nothing
# else :
#     os.mkdir(test_sub_dir)
    

# ####5.2.2 - Images are Not Randomly Saved. Copies images from main folder to respective train,val and test #####
# #Train
# fnames = ['img ({}).jpg'.format(i) for i in range(1,161)] #range(start, stop-1)
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(train_sub_dir, fname)
#     shutil.copyfile(src, dst)
# #Validation
# fnames = ['img ({}).jpg'.format(i) for i in range(161,181)] #range(start, stop-1)
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(validation_sub_dir, fname)
#     shutil.copyfile(src, dst)
# #Test
# fnames = ['img ({}).jpg'.format(i) for i in range(181,201)] #range(start, stop-1)
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_sub_dir, fname)
#     shutil.copyfile(src, dst)

# print('train images: ',len(os.listdir(train_sub_dir)))
# print('val images: ',len(os.listdir(validation_sub_dir)))
# print('test images: ',len(os.listdir(test_sub_dir)))

####Building a convnet#####
from keras import layers
from keras import models

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

#Configure or Compile#
from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])


# ####  5.2.4 Reading Images from the Directories ######
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

# ###### 5.2.8 Fitting the model using a batch generator #####
# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=30,
#       validation_data=validation_generator,
#       validation_steps=50)


# from keras.preprocessing.image import ImageDataGenerator
# #Training of images
# train_datagen = ImageDataGenerator(rescale=1./255)#RESCALE IMAGES
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(150, 150), #RESIZE
#         batch_size=20,
#         class_mode='binary')


# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=5,
#         class_mode='binary')


#COMPILING OR FITTING THE MODEL USING THE BATCH GENERATOR
history = model.fit_generator(
      train_generator,
      steps_per_epoch=3,
      epochs=3,
      validation_data=validation_generator,
      validation_steps=5)



















