
import numpy as np

def submatrix_with_n_numbers(matrix):
    n = 5
    count = 0
    for i in range(len(matrix) - n + 1):
        for j in range(len(matrix[0]) - n + 1):
            submatrix = matrix[i:i+n, j:j+n]
            if np.all(submatrix == submatrix[0, 0]):
                count += 1
    return count
