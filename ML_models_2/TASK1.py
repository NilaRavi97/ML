import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, Reshape, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Sequential
from keras import regularizers
import matplotlib.pyplot as plt

# retrieve the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# scale the data to the range [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a third dimension to the images
# reshaping the arrays to have dimensions (num_samples, height, width, channels)
# channel 1 for gray scale
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Build the model1 from 3c
model1 = Sequential()
model1.add(Reshape((28, 28, 1)))
# Add a convolutional layer with 64 filters and a kernel size of 3x3
# Adding l1 regularization
model1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# Add a max pooling layer
model1.add(MaxPooling2D(pool_size=(2, 2)))
# Add a convolutional layer with 128 filters and a kernel size of 3x3
# Adding l1 regularization
model1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# Add a max pooling layer
model1.add(MaxPooling2D(pool_size=(2, 2)))
# Add a Flatten layer
model1.add(Flatten())
# Add a dense layer with 64 neurons
# Adding l1 regularization
model1.add(Dense(64, activation='relu'))
# Add a dense layer with 10 neurons with softmax activation
model1.add(Dense(10, activation='softmax'))

# Compile the model
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history1 = model1.fit(x_train, y_train, epochs=5)

# Evaluate the model on the training set
_, train_accuracy = model1.evaluate(x_train, y_train)
print('model1 Train accuracy:', train_accuracy)

test_loss, test_accuracy = model1.evaluate(x_test, y_test)
print('model1 Test accuracy:', test_accuracy)

# Build the model2 with dropout layer
model2 = Sequential()
model2.add(Reshape((28, 28, 1)))
# Add a convolutional layer with 64 filters and a kernel size of 3x3
# Adding l1 regularization
model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# Add a max pooling layer
model2.add(MaxPooling2D(pool_size=(2, 2)))
# Add a convolutional layer with 128 filters and a kernel size of 3x3
# Adding l1 regularization
model2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# Add a max pooling layer
model2.add(MaxPooling2D(pool_size=(2, 2)))
# Add a Flatten layer
model2.add(Flatten())
# Add a dense layer with 64 neurons
# Adding l1 regularization
model2.add(Dense(64, activation='relu'))
# Add a dropout layer with rate of 0.25
# The dropout rate is set to 0.25, which means that during training, 25% of the neurons will be randomly dropped out.
# This prevents overfitting and makes the model more robust to unseen data.
model2.add(Dropout(0.25))
# Add a dense layer with 10 neurons with softmax activation
model2.add(Dense(10, activation='softmax'))

# Compile the model
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model

history2 = model2.fit(x_train, y_train, epochs=5)

# Evaluate the model on the training set
_, train_accuracy = model2.evaluate(x_train, y_train)
print('model2 Train accuracy:', train_accuracy)

test_loss, test_accuracy = model2.evaluate(x_test, y_test)
print('model2 Test accuracy:', test_accuracy)

# Build the model3 with batch normalization
model3 = Sequential()
model3.add(Reshape((28, 28, 1)))
# Add a convolutional layer with 64 filters and a kernel size of 3x3
# Adding l1 regularization
model3.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# Add a batch normalization layer
model3.add(BatchNormalization())
# Add a max pooling layer
model3.add(MaxPooling2D(pool_size=(2, 2)))
# Add a convolutional layer with 128 filters and a kernel size of 3x3
# Adding l1 regularization
model3.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# Add a batch normalization layer
model3.add(BatchNormalization())
# Add a max pooling layer
model3.add(MaxPooling2D(pool_size=(2, 2)))
# Add a Flatten layer
model3.add(Flatten())
# Add a dense layer with 64 neurons
# Adding l1 regularization
model3.add(Dense(64, activation='relu'))
# Add a batch normalization layer
model3.add(BatchNormalization())
# Add a dropout layer with rate of 0.25
# The dropout rate is set to 0.25, which means that during training, 25% of the neurons will be randomly dropped out.
# This prevents overfitting and makes the model more robust to unseen data.
model3.add(Dropout(0.25))
# Add a dense layer with 10 neurons with softmax activation
model3.add(Dense(10, activation='softmax'))

# Compile the model
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history3 = model3.fit(x_train, y_train, epochs=5)

# Evaluate the model on the training set
_, train_accuracy = model3.evaluate(x_train, y_train)
print('Model3 Train accuracy:', train_accuracy)

test_loss, test_accuracy = model3.evaluate(x_test, y_test)
print('Model3 Test accuracy:', test_accuracy)
# Build the model4 - with l1 regularization
model4 = Sequential()
model4.add(Reshape((28, 28, 1)))
# Add a convolutional layer with 64 filters and a kernel size of 3x3
# Adding l1 regularization
model4.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.L1(0.01)))
# Add a batch normalization layer
model4.add(BatchNormalization())
# Add a max pooling layer
model4.add(MaxPooling2D(pool_size=(2, 2)))
# Add a convolutional layer with 128 filters and a kernel size of 3x3
# Adding l1 regularization
model4.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.L1(0.01)))
# Add a batch normalization layer
model4.add(BatchNormalization())
# Add a max pooling layer
model4.add(MaxPooling2D(pool_size=(2, 2)))
# Add a Flatten layer
model4.add(Flatten())
# Add a dense layer with 64 neurons
# Adding l1 regularization
model4.add(Dense(64, activation='relu', kernel_regularizer=regularizers.L1(0.01)))
# Add a batch normalization layer
model4.add(BatchNormalization())
# Add a dropout layer with rate of 0.25
# The dropout rate is set to 0.25, which means that during training, 25% of the neurons will be randomly dropped out.
# This prevents overfitting and makes the model more robust to unseen data.
model4.add(Dropout(0.25))
# Add a dense layer with 10 neurons with softmax activation
model4.add(Dense(10, activation='softmax'))

# Compile the model
model4.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history4 = model4.fit(x_train, y_train, epochs=5)

# Evaluate the model on the training set
_, train_accuracy = model4.evaluate(x_train, y_train)
print('Model4 Train accuracy:', train_accuracy)

test_loss, test_accuracy = model4.evaluate(x_test, y_test)
print('Model4 Test accuracy:', test_accuracy)

# Initial Train accuracy: 0.9464166760444641
# Train accuracy after adding dropout layer Train accuracy: 0.9260666370391846
# Train accuracy after adding batch normalization Train accuracy: 0.9320166707038879
# Test accuracy after adding batch normalization Test accuracy: 0.8985999822616577
# Train accuracy after adding batch normalization was not improved.
# Train accuracy after adding l1 regularization Train accuracy: 0.8250333070755005
# Test accuracy after adding l1 regularization Test accuracy: 0.8166000247001648
# The training and testing accuracy of the modified setup, with the L1 regularization,
# is lower than the accuracy of the previous without regularization. even though the accuracy may be lower,
# the regularized model will generalize better and perform better on unseen data.

# Plot the training accuracy and loss for model1
plt.figure()
plt.plot(history1.history['accuracy'], label='model1_accuracy')
plt.plot(history1.history['loss'], label='model1_loss')

# Plot the training accuracy and loss for model2
plt.plot(history2.history['accuracy'], label='model2_accuracy')
plt.plot(history2.history['loss'], label='model2_loss')

# Plot the training accuracy and loss for model3
plt.plot(history3.history['accuracy'], label='model3_accuracy')
plt.plot(history3.history['loss'], label='model3_loss')

# Plot the training accuracy and loss for model4
plt.plot(history4.history['accuracy'], label='model4_accuracy')
plt.plot(history4.history['loss'], label='model4_loss')

plt.title('Training accuracy and loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()
