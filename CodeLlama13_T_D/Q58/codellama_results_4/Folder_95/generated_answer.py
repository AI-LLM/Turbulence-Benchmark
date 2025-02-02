
import numpy as np
def submatrix_with_n_numbers(matrix, n):
    if matrix.shape[0] < n or matrix.shape[1] < n:
        return 0
    
    count = 0
    for i in range(matrix.shape[0] - n + 1):
        for j in range(matrix.shape[1] - n + 1):
            submatrix = matrix[i:i+n, j:j+n].copy()
            if np.all(submatrix == 87):
                count += 1
    return count
