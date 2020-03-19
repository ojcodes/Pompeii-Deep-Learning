#v6
# -*- coding: utf-8 -*-
"""
About: A simple functioning Deep Learning model with trainining and validaion following 5.2 section of Deep Learning with Python
v6: adding drop out in the layer
#Dropout needs to be added
#Data Augmentation needs to be added
"""
import time
import cv2 as cv
import glob
import numpy as np
import os 
import matplotlib.pyplot as plt
# SETTING UP TENSORFLOW AND KERAS
import tensorflow as tf; print('tf_version:',tf.__version__) #tf: 2.1
from tensorflow import keras; print('keras version:',keras.__version__)
from keras import layers; from keras import models; from keras import optimizers
from keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerators

dir_path="C:\\Users\\osama\\Documents\\opus_shuffled" #has all unsorted images
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

# ##DISPLAY AND SAVE IMAGES
# count=1 #counter for running x times or naming images
# for file in imgs: #each file in this case in an image
#     if count<=200: #temp loop to just run 10 images
#         img=cv.imread(file,-1) #create a variable that reads an image one by one (stores in the memory temporairly). 0 for BW. 1 for Color
#         res_img = cv.resize(img, (200, 200))  # w,h
    
#         #cv.imshow('img'+str(count),res_img)# shows img number
#         #cv.imshow(str(file),res_img) #to display the image #. Helps in knowing whether the images have been shuffled
#         if count<=160: #also used for sorting into folders
#             isSaved = cv.imwrite('C:\\Users\\osama\\Documents\\opus_shuffled\\train\\opus\\train_1'+str(count)+'.jpg', res_img)
#             if isSaved:#checks of the image is saved
#                 pass#print('saved image opus_shuf_'+str(count)+'in the train folder')
#         elif count>160 and count<=180:
#             isSaved = cv.imwrite('C:\\Users\\osama\\Documents\\opus_shuffled\\validation\\opus\\val_'+str(count)+'.jpg', res_img)
#             if isSaved:
#                 pass#print('saved image opus_shuf_'+str(count)+'in the val folder')
#         else:
#             isSaved = cv.imwrite('C:\\Users\\osama\\Documents\\opus_shuffled\\test\\opus\\test_'+str(count)+'.jpg', res_img)
#             if isSaved:
#                 pass#print('saved image opus_shuf_'+str(count)+'in the test folder')      
        
#         cv.waitKey(1) #in ms
#         cv.destroyAllWindows()
#         #print (count)
#         count = count+1



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
#Training of images
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
      steps_per_epoch=3,
      epochs=5,
      validation_data=validation_generator,
      validation_steps=3)

#SAVE THE MODEL
#model.save('opus_v4.h5')

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

# ###TEST
img_path = 'C:\\Users\\osama\\Desktop\\DATA 5000\\PROJECT\\code\\images\\opus_shuffled\\test\\test_shuf_194.JPG'
test_img = cv.imread(img_path,-1)
cv.imshow('test',test_img)
# from keras.preprocessing import image
# import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)
plt.imshow(img_tensor[0])
plt.show()

#Listing 5.27. Instantiating a model from an input tensor and a list of output tensors 
from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)


from keras import models
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
####5.4.1 ends here###
































