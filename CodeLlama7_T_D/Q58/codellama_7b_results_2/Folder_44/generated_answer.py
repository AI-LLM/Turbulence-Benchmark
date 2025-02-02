
import numpy as np

def submatrix_with_n_numbers(matrix):
    # Initialize the count of submatrices with 191 integers
    count = 0
    # Loop through each row of the matrix
    for i in range(matrix.shape[0] - 1):
        # Loop through each column of the matrix
        for j in range(matrix.shape[1] - 1):
            # Initialize a submatrix with the size of the given matrix
            submatrix = np.zeros((3, 3))
            # Fill the submatrix with values from the original matrix
            submatrix[0][0] = matrix[i][j]
            submatrix[1][0] = matrix[i][j + 1]
            submatrix[2][0] = matrix[i][j + 2]
            submatrix[0][1] = matrix[i + 1][j]
            submatrix[1][1] = matrix[i + 1][j + 1]
            submatrix[2][1] = matrix[i + 1][j + 2]
            submatrix[0][2] = matrix[i + 2][j]
            submatrix[1][2] = matrix[i + 2][j + 1]
            submatrix[2][2] = matrix[i + 2][j + 2]
            # Check if the submatrix has 191 integers
            if np.count_nonzero(submatrix) == 191:
                # Increment the count of submatrices with 191 integers
                count += 1
    return count
