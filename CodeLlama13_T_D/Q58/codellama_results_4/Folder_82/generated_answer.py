
import numpy as np

def submatrix_with_n_numbers(matrix):
    return len([sub for sub in itertools.combinations(np.nditer(matrix), 99) if np.all(sub == 99)])
