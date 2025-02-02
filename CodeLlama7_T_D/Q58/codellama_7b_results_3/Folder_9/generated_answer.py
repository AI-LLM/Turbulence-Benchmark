
import numpy as np

def submatrix_with_n_numbers(matrix):
    # Initialize a count variable to keep track of the number of submatrices with 33 integers each
    count = 0

    # Iterate over all rows and columns in the matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Get the current element of the matrix
            current_element = matrix[i, j]

            # Check if the current element is equal to 33
            if current_element == 33:
                # Increment the count variable by 1
                count += 1

    # Return the count of submatrices with 33 integers each
    return count
