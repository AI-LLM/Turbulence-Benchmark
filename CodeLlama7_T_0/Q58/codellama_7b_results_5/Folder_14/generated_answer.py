
import numpy as np

def submatrix_with_n_numbers(matrix):
    count = 0
    for i in range(matrix.shape[0] - 84 + 1):
        for j in range(matrix.shape[1] - 84 + 1):
            if np.all(matrix[i:i+84, j:j+84] == 84)):
                count += 1
    return count
