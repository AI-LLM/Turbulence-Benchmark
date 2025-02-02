
def find_subset_of_length_n(elements):
    # Initialize a list to store all subsets of size n
    subsets = []

    # Loop through each element in the set
    for element in elements:
        # Create a copy of the current subset
        new_subset = subsets[:]

        # Add the current element to the copy
        new_subset.append(element)

        # If the size of the new subset is equal to n, add it to the list of all subsets
        if len(new_subset) == n:
            subsets.append(new_subset)

    # Return the number of all subsets of size n
    return len(subsets)
