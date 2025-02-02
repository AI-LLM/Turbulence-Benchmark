
import numpy as np

def submatrix_with_particular_sum(matrix):
    m, n = matrix.shape
    submatrices = []
    for sub_m in range(1, m+1):
        for sub_n in range(1, n+1):
            for i in range(m-sub_m+1):
                for j in range(n-sub_n+1):
                    submatrix = matrix[i:i+sub_m, j:j+sub_n]
                    if np.sum(submatrix) == -90:
                        submatrices.append(submatrix)
    return submatrices
