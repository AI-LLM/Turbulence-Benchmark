
import numpy as np

def submatrix_with_particular_sum(mat):
    # Initialize an empty list to store the submatrices with sum 2
    submatrices = []
    
    # Loop through each row of the matrix
    for i in range(mat.shape[0]):
        # Loop through each column of the matrix
        for j in range(mat.shape[1]):
            # Initialize a temporary matrix to store the submatrix with sum 2
            tmp_mat = np.zeros((3, 3), dtype=int)
            
            # Fill the temporary matrix with the values from the original matrix
            for k in range(3):
                for l in range(3):
                    tmp_mat[k, l] = mat[i + k - 1, j + l - 1]
            
            # Check if the sum of all integers in the temporary matrix is equal to 2
            if np.sum(tmp_mat) == 2:
                # If so, add the temporary matrix to the list of submatrices with sum 2
                submatrices.append(tmp_mat)
    
    # Return the list of submatrices with sum 2
    return submatrices
