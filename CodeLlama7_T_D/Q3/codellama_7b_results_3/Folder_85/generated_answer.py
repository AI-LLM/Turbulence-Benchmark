
def all_pos_ints_inclusive(my_list):
    # Find the first positive index in the list
    first_positive = next((i for i, x in enumerate(my_list) if x > 0)), None)
    # If no positive integers exist in the specified range, return an empty list
    if first_positive is None:
        return []

    # Find the last positive index in the list

    last_positive = next((i for i, x in enumerate(my_list) if x > 0)), None)
    # If no positive integers exist in the specified range, return an empty list
    if last_positive is None:
        return []

    # Return the list of all positive integers from index 6 to index 8, both inclusive

    return [x for x in my_list[first_positive:last_positive + 1] if x > 0]
