
import numpy as np

def submatrix_with_n_numbers(matrix):
    # Initialize the count of submatrices with 15 integers each to 0
    count = 0
    # Loop over all rows and columns of the matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Check if the current element is equal to 15
            if matrix[i, j] == 15:
                # Increment the count of submatrices with 15 integers each by 1
                count += 1

    # Return the count of submatrices with 15 integers each
    return count
