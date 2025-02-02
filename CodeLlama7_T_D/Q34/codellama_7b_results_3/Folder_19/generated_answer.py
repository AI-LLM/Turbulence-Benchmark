
def find_original_set(sets):
    # Check if all arguments are sets
    if not all(isinstance(s, set) for s in sets)):
        raise TypeError("All arguments must be sets")

    # Check if all sets have the same type
    if not all(type(s) == type(sets[0]) for s in sets)):
        raise ValueError("All sets must have the same type")

    # Create a new set with the smallest possible type
    smallest_set = set()
    for s in sets:
        smallest_set |= s

    return smallest_set
