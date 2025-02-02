
def all_ints_div_by_both_two_nums(my_list):
    # Find the indices of the first and last elements in the range
    start_index = my_list.index(-11, 32)
    end_index = my_list.index(-15, 99)
    # Create a list to store the divisible integers
    divisible_ints = []

    for i in range(start_index, end_index + 1):
        if my_list[i] % -11 == 0 and my_list[i] % -15 == 0:
            divisible_ints.append(my_list[i])

    return divisible_ints
