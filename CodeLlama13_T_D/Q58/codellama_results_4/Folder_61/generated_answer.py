
import numpy as np
def submatrix_with_n_numbers(mat):
    def count_submatrices(i, j, k):
        if i == 0 and j == 0 and k == 0:
            return 1
        elif i == 0 and j == 0:
            return count_submatrices(i, j+1, k-1)
        elif i == 0:
            return count_submatrices(i+1, j, k) + count_submatrices(i, j+1, k-1)
        else:
            return count_submatrices(i+1, j, k) + count_submatrices(i, j+1, k-1) + count_submatrices(i-1, j, k) + count_submatrices(i, j-1, k)
    n = len(mat[0])
    m = len(mat)
    return count_submatrices(n//2, m//2, 4)
