
def find_subset_of_length_n(elements, n):
    # Return the number of all subsets of size n in elements
    return sum(1 << i for i in range(len(elements)))
