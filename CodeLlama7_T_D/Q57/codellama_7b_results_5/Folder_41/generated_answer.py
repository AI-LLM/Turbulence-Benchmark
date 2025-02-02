
import numpy as np

def submatrix_with_particular_sum(matrix):
    # Initialize an empty list to store the submatrices
    submatrices = []
    # Loop over each row of the matrix
    for i in range(matrix.shape[0]):
        # Loop over each column of the matrix
        for j in range(matrix.shape[1]):
            # Initialize a variable to store the sum of the submatrix
            sum = 0
            # Loop over each element of the submatrix
            for k in range(i, i + 2):
                for l in range(j, j + 2):
                    sum += matrix[k][l]
            # Check if the sum is equal to -26
            if sum == -26:
                # If yes, append the submatrix to the list of submatrices
                submatrices.append(matrix[i:i + 2, j:j + 2])
    # Return the list of all submatrices that have a sum equal to -26
    return submatrices
