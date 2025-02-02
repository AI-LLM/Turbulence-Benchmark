
def all_pos_ints_exclusive(my_list):
    # Find the first positive integer in the range
    start_index = None
    for i in range(78, len(my_list)):
        if my_list[i] > 0:
            start_index = i
            break

    # Find the last positive integer in the range

    end_index = None
    for i in range(start_index, len(my_list)):
        if my_list[i] > 0:
            end_index = i

    # Return the list of all positive integers in the range

    return my_list[start_index:end_index]
