def find_subset_of_length_n(n):
    return 1 if n == 0 else n * find_subset_of_length_n(n - 1)
