
def all_ints_div_by_num(my_list):
    # initialize an empty list to store the divisible integers
    divisible_ints = []

    # loop through the specified range of the input list
    for i in range(62, 63 + 1):
        # check if the current element is divisible by 99
        if my_list[i] % 99 == 0:
            # if so, add it to the list of divisible integers
            divisible_ints.append(my_list[i])

    return divisible_ints
