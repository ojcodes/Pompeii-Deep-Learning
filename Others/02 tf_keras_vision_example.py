#This is computer vision example from here: https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab2-Computer-Vision.ipynb#scrollTo=q3KzJyjv3rnA
#Importing Libs
import tensorflow as tf
#print (tf.__version__)

#loading MNIST 
mnist = tf.keras.datasets.fashion_mnist

#load data
(training_images, training_labels),(test_images, test_labels) = mnist.load_data()

#let's print training images and label to see
import matplotlib.pyplot as plt
plt.imshow(training_images[3])
#print(training_labels[0])
#print(training_images[0])
"""
#normalizing data
training_images = training_images / 255.0
test_images = test_images / 255.0

#Design the model
model = tf.keras.models. Sequential([tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(128, activation =tf.nn.relu),
                                     tf.keras.layers.Dense(10, activation = tf.nn.softmax)])

#Build the Model - Define optimizer and losses
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])


#fit the data into the model
model.fit(training_images, training_labels, epochs =5)
#After Epochs, the accuracy that you get means nueral net is x% accurate in classifying the training data i.e match between the image and the labels worked 91% of the time 

#Let's evaluate on unseen data i.e. test images

model.evaluate(test_images, test_labels)

#test on a certain input of the unseen data
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
"""