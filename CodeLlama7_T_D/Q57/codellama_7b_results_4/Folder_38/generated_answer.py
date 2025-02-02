
import numpy as np

def submatrix_with_particular_sum(matrix):
    # Initialize an empty list to store the submatrices
    submatrices = []

    # Iterate over each row of the matrix
    for i in range(matrix.shape[0]):
        # Iterate over each column of the matrix
        for j in range(matrix.shape[1]):
            # Initialize a variable to store the sum of the elements in the submatrix
            sum = 0

            # Iterate over each element in the submatrix
            for k in range(i, i + 3):
                for l in range(j, j + 3):
                    # Add the element to the sum
                    sum += matrix[k][l]

                    # If the sum is equal to 34, add the submatrix to the list and break out of the loops
                    if sum == 34:
                        submatrices.append(matrix[i:i + 3][j:j + 3]])
                        break

            # If the sum is equal to 34, add the submatrix to the list and break out of the loops
            if sum == 34:
                submatrices.append(matrix[i:i + 3][j:j + 3]])
                break

    # Return the list of all submatrices with a sum equal to 34
    return submatrices
