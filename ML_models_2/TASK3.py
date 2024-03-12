import tensorflow as tf
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Flatten, Reshape, MaxPooling2D
from keras.models import Sequential


# retrieve the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# scale the data to the range [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a third dimension to the images
# reshaping the arrays to have dimensions (num_samples, height, width, channels)
# channel 1 for gray scale
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Build the model
model = Sequential()
model.add(Reshape((28, 28, 1)))
# Add a convolutional layer with 64 filters and a kernel size of 3x3
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# Add a convolutional layer with 128 filters and a kernel size of 3x3
model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
# Add a Flatten layer
model.add(Flatten())
# Add a dense layer with 64 neurons
model.add(Dense(64, activation='relu'))
# Add a dense layer with 10 neurons with softmax activation
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the training set
_, train_accuracy = model.evaluate(x_train, y_train)
print('Train accuracy:', train_accuracy)

# Build the model
model1 = Sequential()
model1.add(Reshape((28, 28, 1)))
# Add a convolutional layer with 64 filters and a kernel size of 3x3
model1.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# Add a max pooling layer
model1.add(MaxPooling2D(pool_size=(2,2)))
# Add a convolutional layer with 128 filters and a kernel size of 3x3
model1.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
# Add a max pooling layer
model1.add(MaxPooling2D(pool_size=(2,2)))
# Add a Flatten layer
model1.add(Flatten())
# Add a dense layer with 64 neurons
model1.add(Dense(64, activation='relu'))
# Add a dense layer with 10 neurons with softmax activation
model1.add(Dense(10, activation='softmax'))

# Compile the model
model1.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Train the model
model1.fit(x_train, y_train, epochs=5)

# Evaluate the model on the training set
_, train_accuracy = model1.evaluate(x_train, y_train)
print('Train accuracy:', train_accuracy)

#The train accuracy of the model decreased after adding max-pooling layers. This could be due to overfitting prevention,
#as max-pooling reduces the size of the feature maps and eliminates less significant information. However, if the model was
#already underfitting the training data (i.e., it had high bias), adding max-pooling may have further decreased the
#model's ability to fit the training data, resulting in poorer performance