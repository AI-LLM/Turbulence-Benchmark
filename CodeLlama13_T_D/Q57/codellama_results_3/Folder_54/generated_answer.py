
import numpy as np

def submatrix_with_particular_sum(matrix):
    # Initialize an empty list to store the found submatrices
    submatrices = []
    
    # Loop through each row of the matrix
    for i in range(matrix.shape[0]):
        # Loop through each column of the matrix
        for j in range(matrix.shape[1]):
            # Initialize a temporary submatrix with the current element
            temp_submatrix = np.array([[matrix[i,j]]])
            
            # Check if the sum of all integers in the temporary submatrix equals -59
            if temp_submatrix.sum() == -59:
                # If so, add the temporary submatrix to the list of found submatrices
                submatrices.append(temp_submatrix)
            
            # Loop through each row and column after the current element
            for k in range(i+1, matrix.shape[0]):
                for l in range(j+1, matrix.shape[1]):
                    # Add the elements of the temporary submatrix with the current element
                    temp_submatrix = np.append(temp_submatrix, [[matrix[k,l]]], axis=0)
                    
                    # Check if the sum of all integers in the updated temporary submatrix equals -59
                    if temp_submatrix.sum() == -59:
                        # If so, add the updated temporary submatrix to the list of found submatrices
                        submatrices.append(temp_submatrix)
    
    return submatrices
