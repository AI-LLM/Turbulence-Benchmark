
def all_pos_ints_exclusive(my_list):
    # Find the first positive integer in the list
    for i in range(32, len(my_list)):
        if my_list[i] > 0:
            start_index = i
            break

    else:

        return []

    # Find the last positive integer in the list

    for i in range(start_index, len(my_list)):
        if my_list[i] > 0:
            end_index = i

    # Return the list of all positive integers between the start and end indices, exclusive

    return [my_list[i] for i in range(start_index + 1, end_index + 1)]
