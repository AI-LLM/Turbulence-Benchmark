import numpy as np
def submatrix_with_particular_sum(matrix):
    return [submatrix for row in range(len(matrix)) for submatrix in range(row, len(matrix)) if sum(matrix[row:submatrix]) == -77]
