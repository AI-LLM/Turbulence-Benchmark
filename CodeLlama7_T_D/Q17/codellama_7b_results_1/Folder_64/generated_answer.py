
def all_ints_div_by_both_two_nums(my_list):
    # initialize an empty list to store the divisible integers
    divisible_ints = []

    # loop through the list of integers

    for i in range(len(my_list)):

        # check if the current integer is divisible by both 11 and -7

        if my_list[i] % 11 == 0 and my_list[i] % -7 == 0:

            # if it is, add it to the list of divisible integers

            divisible_ints.append(my_list[i])

    # return the list of all integers divisible by both 11 and -7 from index 0 to index 3, both inclusive

    return divisible_ints
