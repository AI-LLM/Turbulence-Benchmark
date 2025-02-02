
import numpy as np

def submatrix_with_n_numbers(matrix):
    # Initialize the count variable
    count = 0

    # Iterate over each row of the matrix
    for i in range(matrix.shape[0]):
        # Get the current row of the matrix
        row = matrix[i]

        # Iterate over each column of the matrix
        for j in range(matrix.shape[1]):
            # Get the current column of the matrix
            col = matrix[:, j]

            # Check if the submatrix has 166 integers
            if np.count_nonzero(row) == 166 and np.count_nonzero(col) == 166:
                # Increment the count variable
                count += 1

    # Return the count of all submatrices with 166 integers each
    return count
