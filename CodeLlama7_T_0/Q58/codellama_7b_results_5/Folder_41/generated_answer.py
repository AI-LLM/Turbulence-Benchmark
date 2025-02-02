
import numpy as np

def submatrix_with_n_numbers(matrix):
    count = 0
    for i in range(matrix.shape[0] - 72):
        for j in range(matrix.shape[1] - 72):
            if matrix[i:i+73, j:j+73].sum() == 73 * 73:
                count += 1

    return count
