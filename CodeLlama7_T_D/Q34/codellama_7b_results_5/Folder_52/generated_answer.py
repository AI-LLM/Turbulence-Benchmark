
def find_original_set(sets):
    # Sort the sets in ascending order based on their length
    sorted_sets = sorted(sets, key=len)
    # Initialize the smallest set as the first element in the list
    smallest_set = sorted_sets[0]
    # Iterate through the remaining sets and find the smallest set that contains all of them
    for i in range(1, len(sorted_sets)):
        current_set = sorted_sets[i]
        if all(element in smallest_set for element in current_set):
            smallest_set = current_set
    return smallest_set
