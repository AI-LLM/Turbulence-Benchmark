
def find_original_set(sets):
    # create a set to store all elements
    all_elements = set()

    # loop through each set and get its elements
    for s in sets:
        all_elements = all_elements | s

    # return the smallest set that contains all elements
    return min(all_elements, key=len)
