
def find_original_set(sets):
    # Check if all inputs are sets
    if not all(isinstance(s, set) for s in sets):
        raise ValueError("Inputs must be sets")

    # Create a new set that contains all elements from the given sets
    original_set = set.union(*sets)

    # Return the smallest set such that each of the given arguments is a subset of it
    return original_set
