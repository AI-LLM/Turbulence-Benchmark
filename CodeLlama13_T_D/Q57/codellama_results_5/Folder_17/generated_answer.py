
import numpy as np

def submatrix_with_particular_sum(matrix):
    # Initialize an empty list to store the results
    result = []
    
    # Loop through each row of the matrix
    for i in range(matrix.shape[0]):
        # Loop through each column of the matrix
        for j in range(matrix.shape[1]):
            # Initialize a temporary submatrix with the current element
            temp_submatrix = np.array([[matrix[i,j]]])
            
            # Check if the sum of all integers in the temporary submatrix is equal to 398
            if np.sum(temp_submatrix) == 398:
                # If it is, add the temporary submatrix to the result list
                result.append(temp_submatrix)
            
            # Loop through each row of the temporary submatrix
            for k in range(1, temp_submatrix.shape[0]):
                # Add the next row to the temporary submatrix
                temp_submatrix = np.vstack((temp_submatrix, matrix[i+k,j]))
                
                # Check if the sum of all integers in the temporary submatrix is equal to 398
                if np.sum(temp_submatrix) == 398:
                    # If it is, add the temporary submatrix to the result list
                    result.append(temp_submatrix)
    
    return result
