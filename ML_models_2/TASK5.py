import numpy as np
from scipy.signal import convolve2d
from sklearn.metrics import pairwise_distances


def convolution(input_matrix, kernel_matrix):
    return convolve2d(input_matrix, kernel_matrix, mode='full')


def max_pooling(input_matrix, pool_size):
    inputrow, inputcol = input_matrix.shape
    output_rows, output_cols = inputrow//pool_size, inputcol//pool_size
    output_matrix = np.zeros((output_rows, output_cols))
    for i in range(output_rows):
        for j in range(output_cols):
            row_start, col_start = i*pool_size, j*pool_size
            row_end, col_end = row_start+pool_size, col_start+pool_size
            output_matrix[i, j] = np.max(input_matrix[row_start:row_end, col_start:col_end])
    return output_matrix


# Test for convolution
input_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
kernel_matrix = np.array([[1, 0], [0, 1]])
print('convolution', convolution(input_matrix, kernel_matrix))

# Test for max-pooling
input_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
pool_size = 2
print('max pooling', max_pooling(input_matrix, pool_size))