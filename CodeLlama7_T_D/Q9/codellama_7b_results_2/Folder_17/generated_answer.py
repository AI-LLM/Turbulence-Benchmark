
def all_odd_ints_inclusive(my_list):
    # Get the indices of the first and last odd integers in the list
    first_odd_index = next((i for i, x in enumerate(my_list) if x % 2 == 1), None)
    last_odd_index = next((i for i, x in enumerate(my_list) if x % 2 == 1), None)
    # If no odd integers exist in the specified range, return an empty list
    if first_odd_index is None or last_odd_index is None:
        return []

    # Return the list of all odd integers from index 62 to index 78, both inclusive
    return [my_list[i] for i in range(first_odd_index, last_odd_index + 1)]
