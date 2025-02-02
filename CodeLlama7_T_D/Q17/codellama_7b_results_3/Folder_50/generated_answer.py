
def all_ints_div_by_both_two_nums(my_list):
    # Initialize an empty list to store the divisible integers
    div_ints = []

    # Iterate through the specified range of indices in the input list
    for i in range(46, 92):
        # Check if the current integer is divisible by both 67 and 15
        if my_list[i] % 67 == 0 and my_list[i] % 15 == 0:
            # If it is, add it to the list of divisible integers
            div_ints.append(my_list[i])

    # Return the list of all integers divisible by both 67 and 15 from index 46 to index 91, both inclusive
    return div_ints
