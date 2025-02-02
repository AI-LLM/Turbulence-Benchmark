
import numpy as np

def submatrix_with_n_numbers(matrix):
    # Initialize the count to 0
    count = 0
    # Loop over each row of the matrix
    for i in range(len(matrix)):
        # Loop over each column of the matrix
        for j in range(len(matrix[i])):
            # Initialize a variable to store the submatrix
            submatrix = []

            # Check if the current element is 111
            if matrix[i][j] == 111:
                # If it is, add it to the submatrix
                submatrix.append(matrix[i][j])

                # Loop over each row of the submatrix
                for k in range(len(submatrix)):
                    # Loop over each column of the submatrix
                    for l in range(len(submatrix[k])):
                        # Check if the current element is 111
                        if submatrix[k][l] == 111:
                            # If it is, add it to the submatrix
                            submatrix.append(submatrix[k][l])

                # Increment the count by 1 for each submatrix found
                count += len(submatrix)

    # Return the count of all submatrices containing 111 integers
    return count
