
import numpy as np

def submatrix_with_particular_sum(matrix):
    # Initialize the list of submatrices
    submatrices = []

    # Iterate over each row and column of the matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Get the submatrix of size 3x3 starting at position (i, j)
            submatrix = matrix[i:i+3, j:j+3]

            # Check if the sum of all integers in the submatrix is equal to 752
            if np.sum(submatrix) == 752:
                # If it is, add it to the list of submatrices
                submatrices.append(submatrix)

    # Return the list of submatrices that have a sum of all integers equal to 752
    return submatrices
