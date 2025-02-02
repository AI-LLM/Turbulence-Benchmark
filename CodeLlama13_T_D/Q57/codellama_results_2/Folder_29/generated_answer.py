
import numpy as np

def submatrix_with_particular_sum(matrix):
    # Initialize an empty list to store the submatrices
    submatrices = []
    
    # Loop through each row of the matrix
    for i in range(matrix.shape[0]):
        # Loop through each column of the matrix
        for j in range(matrix.shape[1]):
            # Initialize a temporary list to store the submatrix
            temp_list = []
            
            # Loop through each row of the submatrix
            for k in range(i, matrix.shape[0]):
                # Loop through each column of the submatrix
                for l in range(j, matrix.shape[1]):
                    # Add the current element to the temporary list
                    temp_list.append(matrix[k][l])
                    
            # Check if the sum of all integers in the submatrix is equal to 84
            if sum(temp_list) == 84:
                # If it is, add the submatrix to the list of submatrices
                submatrices.append(np.array(temp_list).reshape(-1, matrix.shape[1]))
    
    # Return the list of all submatrices with a particular sum
    return submatrices
