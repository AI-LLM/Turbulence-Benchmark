
import numpy as np

def submatrix_with_particular_sum(matrix):
    submatrices = []
    m, n = matrix.shape
    for i in range(m):
        for j in range(n):
            for k in range(i+1, m+1):
                for l in range(j+1, n+1):
                    submatrix = matrix[i:k, j:l]
                    if np.sum(submatrix) == -128:
                        submatrices.append(submatrix.tolist())
    return submatrices
