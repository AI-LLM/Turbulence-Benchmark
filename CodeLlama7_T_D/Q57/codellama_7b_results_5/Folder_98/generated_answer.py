
import numpy as np
def submatrix_with_particular_sum(matrix):
    # Initialize a list to store the submatrices
    submatrices = []
    # Loop over each row of the matrix
    for i in range(matrix.shape[0]):
        # Loop over each column of the matrix
        for j in range(matrix.shape[1]):
            # Extract a submatrix from the original matrix
            submatrix = matrix[i:, j:]
            # Check if the sum of all integers in the submatrix is equal to 0
            if np.sum(submatrix) == 0:
                # If the condition is met, add the submatrix to the list
                submatrices.append(submatrix)
    # Return the list of submatrices that satisfy the condition
    return submatrices
