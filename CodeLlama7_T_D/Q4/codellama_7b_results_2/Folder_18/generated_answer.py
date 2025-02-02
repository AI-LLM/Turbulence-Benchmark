
def all_pos_ints_exclusive(my_list):
    # Find the index of the first positive integer in the list
    first_pos_int = next((i for i, x in enumerate(my_list) if x > 0)), None)
    # If there are no positive integers in the list, return an empty list
    if first_pos_int is None:
        return []

    # Find the index of the last positive integer in the list

    last_pos_int = next((i for i, x in enumerate(my_list[first_pos_int:]) if x > 0)), None)
    # If there are no positive integers in the list, return an empty list
    if last_pos_int is None:
        return []

    # Return the list of all positive integers from index 57 to index 84, both exclusive
    return my_list[first_pos_int + 57:last_pos_int + 84]
