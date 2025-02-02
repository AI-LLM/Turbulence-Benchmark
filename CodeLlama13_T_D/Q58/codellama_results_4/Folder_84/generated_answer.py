
import numpy as np

def submatrix_with_n_numbers(matrix):
    # Initialize a counter for the number of submatrices with 127 integers
    count = 0
    
    # Loop through each row in the matrix
    for i in range(matrix.shape[0]):
        # Loop through each column in the matrix
        for j in range(matrix.shape[1]):
            # If the current element is equal to 127, increment the counter
            if matrix[i,j] == 127:
                count += 1
    
    # Return the final count of submatrices with 127 integers
    return count
