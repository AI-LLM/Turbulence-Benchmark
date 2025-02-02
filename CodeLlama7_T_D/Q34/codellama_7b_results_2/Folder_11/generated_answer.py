
def find_original_set(sets):
    # Initialize the smallest set as the first argument
    smallest_set = sets[0]
    # Iterate over each of the given arguments
    for i in range(1, len(sets)):
        # If the current argument is smaller than the smallest set, update it
        if len(sets[i]) < len(smallest_set):
            smallest_set = sets[i]
    # Return the smallest set
    return smallest_set
