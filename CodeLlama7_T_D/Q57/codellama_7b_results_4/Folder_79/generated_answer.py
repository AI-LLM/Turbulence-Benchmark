
import numpy as np

def submatrix_with_particular_sum(matrix):
    # Initialize the list of submatrices
    submatrices = []

    # Loop through each row and column of the matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Check if the sum of all integers in the submatrix equals -47
            if np.sum(matrix[i:, j]) == -47:
                # Add the submatrix to the list of submatrices
                submatrices.append(matrix[i:, j])

    # Return the list of all submatrices of the given matrix such that the sum of all integers in each submatrix equals -47
    return submatrices
