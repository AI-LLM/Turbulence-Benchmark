
def all_odd_ints_exclusive(my_list):
    # Find the index of the first odd integer in the list
    first_odd_index = None
    for i in range(len(my_list)):
        if my_list[i] % 2 != 0:
            first_odd_index = i
            break

    # If no odd integers exist in the specified range, return an empty list
    if first_odd_index is None:
        return []

    # Return the list of all odd integers from index 323 to index 972, both exclusive
    return my_list[first_odd_index + 1:972:2]
