import keras
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import LSTM
from keras.utils import to_categorical

# Fetch the data
X, y = fetch_openml('eeg-eye-state', version=1, return_X_y=True)

# Scale the data to [0,1]
X = X / np.max(X)

# Split the data into training and test sets with a test ratio of 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Create sliding windows for training and test data with a sequence length of 10
sequence_length = 10
train_generator = TimeseriesGenerator(X_train, y_train, length=sequence_length, batch_size=32)
test_generator = TimeseriesGenerator(X_test, y_test, length=sequence_length, batch_size=32)

model = keras.Sequential()

model.add(keras.layers.InputLayer(input_shape=(sequence_length, X_train.shape[1])))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(128, activation='relu')))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(256, activation='relu')))
model.add(keras.layers.SimpleRNN(128, return_sequences=True))
model.add(keras.layers.SimpleRNN(128))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(3, activation='softmax'))



# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
# 10 percent of the training data is used for validation
history = model.fit(train_generator, validation_data=test_generator, epochs=15, validation_split=0.1)

test_loss, test_acc = model.evaluate(test_generator)
print("Test accuracy:", test_acc)

model1 = keras.Sequential()

model1.add(keras.layers.InputLayer(input_shape=(sequence_length, X_train.shape[1])))
model1.add(keras.layers.TimeDistributed(keras.layers.Dense(128, activation='relu')))
model1.add(keras.layers.TimeDistributed(keras.layers.Dense(256, activation='relu')))
model1.add(keras.layers.LSTM(128, return_sequences=True))
model1.add(keras.layers.LSTM(128))
model1.add(keras.layers.Dense(128, activation='relu'))
model1.add(keras.layers.Dropout(0.5))
model1.add(keras.layers.Dense(64, activation='relu'))
model1.add(keras.layers.Dropout(0.5))
model1.add(keras.layers.Dense(3, activation='softmax'))

# Compile the model
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
# 10 percent of the training data is used for validation
history1 = model1.fit(train_generator, validation_data=test_generator, epochs=15, validation_split=0.1)

test_loss1, test_acc1 = model1.evaluate(test_generator)
print("Test accuracy for LSTM:", test_acc1)


