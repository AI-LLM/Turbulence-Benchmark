
import numpy as np

def submatrix_with_n_numbers(matrix):
    submatrix_count = 0
    rows, cols = matrix.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            if matrix[i, j] == 46:
                if np.all(matrix[i:i+2, j:j+2] == 46):
                    submatrix_count += 1
    return submatrix_count
