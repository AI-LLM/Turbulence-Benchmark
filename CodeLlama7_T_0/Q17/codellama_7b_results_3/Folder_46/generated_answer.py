
def all_ints_div_by_both_two_nums(my_list):
    # Initialize an empty list to store the divisible integers
    divisible_ints = []

    # Iterate through the list of integers from index 11 to index 46, both inclusive
    for i in range(11, 47):
        # Check if the current integer is divisible by both 55 and 36
        if my_list[i] % 55 == 0 and my_list[i] % 36 == 0:
            # If it is, add it to the list of divisible integers
            divisible_ints.append(my_list[i])

    # Return the list of all integers divisible by both 55 and 36 from index 11 to index 46, both inclusive
    return divisible_ints
