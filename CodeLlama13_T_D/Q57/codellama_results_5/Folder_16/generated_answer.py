
import numpy as np

def submatrix_with_particular_sum(matrix):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if np.sum(matrix[i:j]) == -157:
                return matrix[i:j]
    return []
