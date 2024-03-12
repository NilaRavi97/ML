import tensorflow as tf
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# retrieve the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# scale the data to the range [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

indices = np.random.choice(len(x_train), 5, replace=False)

# Define the convolutions
conv1 = [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]
conv2 = [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]
conv3 = [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]

# Loop over the selected indices
for i in indices:
    # Extract the image and apply convolution
    image = x_train[i]
    convolved1 = ndimage.convolve(image, conv1)
    convolved2 = ndimage.convolve(image, conv2)
    convolved3 = ndimage.convolve(image, conv3)

    # Create a figure with subplots
    fig, ax = plt.subplots(1, 4, figsize=(10, 3))
    ax[0].imshow(image, cmap='gray')
    ax[1].imshow(convolved1, cmap='gray')
    ax[2].imshow(convolved2, cmap='gray')
    ax[3].imshow(convolved3, cmap='gray')
    plt.show()