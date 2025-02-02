
import numpy as np

def submatrix_with_particular_sum(matrix):
    submatrices = []
    rows, columns = matrix.shape
    
    for top in range(rows):
        for bottom in range(top+1, rows+1):
            for left in range(columns):
                for right in range(left+1, columns+1):
                    submatrix = matrix[top:bottom, left:right]
                    if np.sum(submatrix) == -47:
                        submatrices.append(submatrix.tolist())
    
    return submatrices
