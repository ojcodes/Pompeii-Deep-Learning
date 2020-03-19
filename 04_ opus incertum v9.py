"""
About: A simple functioning Deep Learning model with trainining and validaion following 5.2 until 5.4.1 section of Deep Learning with Python
#Dropout needs to be added
#Data Augmentation needs to be added
#Though it shows what is going on with the image. And accuracy is too good to be true between train and
#To do: plot testing and plot an image with an ROI on the testing image
"""


# SETTING UP TENSORFLOW AND KERAS
import sys;print('python version:',sys.version)
import tensorflow as tf; print('tf_version:',tf.__version__) #tf: 2.1
from tensorflow import keras; print('keras version:',keras.__version__)
from tensorflow.python.platform import build_info as tf_build_info
print('CUDA Version:',tf_build_info.cuda_version_number)
print('CUDNN Version:',tf_build_info.cudnn_version_number)


import os, shutil
original_dataset_dir="C:\\Users\\osama\\Documents\\opus_shuffled3" #has all unsorted images

####Create train, validation and test folders and sub-folders. Keras needs sub-folders:https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# Create a Test Dir##### #create a folder "train": os.path.join gets path and the name of the new directory#os.mkdir(directory_name_path) makes it#You have to error check whether the folder already exist
train_dir = os.path.join(original_dataset_dir, 'train') 
if os.path.exists(train_dir):
    pass #does nothing
else :
    os.mkdir(train_dir)

train_sub_dir = os.path.join(train_dir, 'train_sub') 
if os.path.exists(train_sub_dir):
    pass #does nothing
else :
    os.mkdir(train_sub_dir)
    
# #####Create a Validation Dir#####
validation_dir = os.path.join(original_dataset_dir, 'validation') 
if os.path.exists(validation_dir):
    pass #does nothing
else :
    os.mkdir(validation_dir)

validation_sub_dir = os.path.join(validation_dir, 'validation_sub') 
if os.path.exists(validation_sub_dir):
    pass #does nothing
else :
    os.mkdir(validation_sub_dir)

# ####Create a Test Dir#####
test_dir = os.path.join(original_dataset_dir, 'test') 
if os.path.exists(test_dir):
    pass #does nothing
else :
    os.mkdir(test_dir)

test_sub_dir = os.path.join(test_dir, 'test_sub') 
if os.path.exists(test_sub_dir):
    pass #does nothing
else :
    os.mkdir(test_sub_dir)
    

####5.2.2 - Images are Not Randomly Saved. Copies images from main folder to respective train,val and test #####
#Train
fnames = ['img ({}).jpg'.format(i) for i in range(1,161)] #range(start, stop-1)
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_sub_dir, fname)
    shutil.copyfile(src, dst)
#Validation
fnames = ['img ({}).jpg'.format(i) for i in range(161,181)] #range(start, stop-1)
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_sub_dir, fname)
    shutil.copyfile(src, dst)
#Test
fnames = ['img ({}).jpg'.format(i) for i in range(181,201)] #range(start, stop-1)
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_sub_dir, fname)
    shutil.copyfile(src, dst)

print('train images: ',len(os.listdir(train_sub_dir)))
print('val images: ',len(os.listdir(validation_sub_dir)))
print('test images: ',len(os.listdir(test_sub_dir)))

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


#####  5.2.4 Reading Images from the Directories ######
# from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         class_mode='binary')

# ###### 5.2.8 Fitting the model using a batch generator #####
# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=30,
#       validation_data=validation_generator,
#       validation_steps=50)


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
        batch_size=20,
        class_mode='binary')


#COMPILING OR FITTING THE MODEL USING THE BATCH GENERATOR
history = model.fit_generator(
      train_generator,
      steps_per_epoch=3,
      epochs=3,
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

##### TEST #####
import matplotlib.image as mpimg
#Give the right image path. Seems like \\ is required for the path in python
img_path = 'C:\\Users\\osama\\Documents\\opus_shuffled3\\img (194).JPG'
a= mpimg.imread(img_path)
plt.imshow(a)

from keras.preprocessing import image
import numpy as np

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
































