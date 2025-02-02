
import numpy as np

def submatrices_with_particular_sum(matrix):
    # Initialize an empty list to store the submatrices
    submatrices = []
    
    # Iterate over each row of the matrix
    for i in range(matrix.shape[0]):
        # Iterate over each column of the matrix
        for j in range(matrix.shape[1]):
            # Initialize a temporary matrix to store the current submatrix
            temp_matrix = np.zeros((i+1, j+1))
            
            # Fill the temporary matrix with the values from the original matrix
            for k in range(i+1):
                for l in range(j+1):
                    temp_matrix[k][l] = matrix[k][l]
            
            # Check if the sum of all integers in the temporary matrix is 13
            if np.sum(temp_matrix) == 13:
                # If it is, add the temporary matrix to the list of submatrices
                submatrices.append(temp_matrix)
    
    return submatrices
