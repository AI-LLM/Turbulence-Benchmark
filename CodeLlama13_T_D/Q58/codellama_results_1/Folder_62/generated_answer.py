
import numpy as np

def submatrix_with_n_numbers(matrix):
    # Initialize a counter for the number of submatrices with 186 integers
    count = 0
    
    # Iterate over each row and column of the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # Initialize a temporary matrix to store the current submatrix
            temp_mat = np.zeros((3, 3), dtype=int)
            
            # Fill the temporary matrix with the values from the original matrix
            for k in range(3):
                for l in range(3):
                    temp_mat[k, l] = matrix[i + k][j + l]
                    
            # Check if the temporary matrix has 186 integers
            if np.sum(temp_mat) == 186:
                count += 1
                
    return count
