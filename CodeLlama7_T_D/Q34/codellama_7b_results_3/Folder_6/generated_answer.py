
def find_original_set(sets):
    # Sort the sets in increasing order of their sizes
    sorted_sets = sorted(sets, key=len)
    # Initialize the smallest set as the first element of the list
    smallest_set = sorted_sets[0]
    # Iterate over the remaining elements of the list
    for i in range(1, len(sorted_sets)):
        # If the current set is a subset of the smallest set, then we do not need to update the smallest set
        if sorted_sets[i].issubset(smallest_set):
            continue

        # Otherwise, check if the current set is a subset of any element in the smallest set
        for j in range(len(smallest_set)):
            # If the current set is a subset of any element in the smallest set, then we do not need to update the smallest set
            if sorted_sets[i].issubset(smallest_set[j]):
                break

        # If the current set is not a subset of any element in the smallest set, then we need to update the smallest set with the current set
        else:
            smallest_set = sorted_sets[i]
    return smallest_set
