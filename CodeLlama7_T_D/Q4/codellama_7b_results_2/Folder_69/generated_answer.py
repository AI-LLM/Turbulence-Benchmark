
def all_pos_ints_exclusive(my_list):
    # create a new list for the output
    result = []

    # loop through the input list and check if each element is positive
    for i in range(len(my_list)):
        if my_list[i] > 0:
            # add the positive integer to the output list
            result.append(my_list[i])

    # return the output list
    return result
