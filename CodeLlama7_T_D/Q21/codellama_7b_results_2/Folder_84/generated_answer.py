
def all_ints_div_by_num(my_list):
    # initialize an empty list to store the results
    result = []

    # loop through the input list
    for i in range(52, 54):
        # check if the current element is divisible by 93
        if my_list[i] % 93 == 0:
            # if it is, add it to the result list
            result.append(my_list[i])

    # return the result list
    return result
