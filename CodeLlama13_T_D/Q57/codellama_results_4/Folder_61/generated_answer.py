
import numpy as np

def submatrix_with_particular_sum(matrix):
    # Initialize an empty list to store the submatrices with sum 8
    submatrices = []
    
    # Loop through each row of the matrix
    for i in range(matrix.shape[0]):
        # Loop through each column of the matrix
        for j in range(matrix.shape[1]):
            # Initialize a temporary list to store the submatrix elements
            temp_list = []
            
            # Loop through each row and column of the submatrix
            for k in range(i, matrix.shape[0]):
                for l in range(j, matrix.shape[1]):
                    # Add the element to the temporary list if it is within the bounds of the matrix
                    if 0 <= k < matrix.shape[0] and 0 <= l < matrix.shape[1]:
                        temp_list.append(matrix[k, l])
                    
            # Check if the sum of the temporary list is equal to 8
            if np.sum(temp_list) == 8:
                # If so, add the submatrix to the list of submatrices
                submatrices.append(np.array(temp_list).reshape((k - i + 1, l - j + 1)))
    
    return submatrices
