
import numpy as np

def submatrix_with_n_numbers(matrix):
    # Initialize the count to 0
    count = 0
    # Iterate over all rows and columns of the matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Check if the current submatrix has 67 integers
            if sum(matrix[i:, j]) == 67:
                # If it does, increment the count
                count += 1

    # Return the count of all submatrices with 67 integers
    return count
