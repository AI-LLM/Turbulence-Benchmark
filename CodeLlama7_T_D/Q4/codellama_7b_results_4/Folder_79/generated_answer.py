
def all_pos_ints_exclusive(my_list):
    # Find the first positive integer in the list
    first_pos_int = next((x for x in my_list if x > 0), None)
    # If no positive integers exist in the specified range, return an empty list
    if first_pos_int is None:
        return []

    # Return the list of all positive integers from index 3 to index 8, both exclusive
    return [x for x in my_list[first_pos_int - 1 : 8] if x > 0]
