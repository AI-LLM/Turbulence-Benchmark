
import numpy as np
def submatrix_with_particular_sum(matrix):
    # Initialize the list of submatrices with sum -81
    submatrices = []
    # Loop through each row of the matrix
    for i in range(matrix.shape[0]):
        # Loop through each column of the matrix
        for j in range(matrix.shape[1]):
            # Initialize a variable to store the sum of the submatrix
            sum = 0
            # Loop through each element of the submatrix
            for row in range(i, i + 3):
                for col in range(j, j + 3):
                    # Add the value of the element to the sum
                    sum += matrix[row][col]
            # Check if the sum is equal to -81
            if sum == -81:
                # If it is, add the submatrix to the list of submatrices
                submatrices.append([[matrix[i][j]], [matrix[i + 1][j]], [matrix[i + 2][j]]], [[matrix[i][j + 1]], [matrix[i + 1][j + 1]], [matrix[i + 2][j + 1]]], [[matrix[i][j + 2]], [matrix[i + 1][j + 2]], [matrix[i + 2][j + 2]]]))
    # Return the list of submatrices with sum -81
    return submatrices
