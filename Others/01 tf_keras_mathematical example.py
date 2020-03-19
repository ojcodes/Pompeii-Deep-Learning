#Importing Libs
import tensorflow as tf
import numpy as np

#importing keras API from tensorflow
from tensorflow import keras

#video link :https://www.youtube.com/watch?v=KNAWp2S3w94
#Defining Model
#import model//It has 1 layer, and that layer has 1 neuron, and the input shape to it is just 1 value.
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
#defining loss function and optimizer
#loss function = measure of guessed answer vs the correct answer
#optimizer = it tried to minimize the loss function
#for loss function, we use mean_squared_error, while for optimizer, we use sigmoid


#input and output
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)


#fit given data to the model and specify epochs
model.fit(xs,ys,epochs=100)

#predict answers for unseen data/input
print(model.predict([10.0]))
