
def sum_ints_div_by_either_nums(my_list):
    # Initialize the sum to 0
    sum = 0

    # Iterate over the list from index 25 to index 70, both inclusive
    for i in range(25, 71):

        # Check if the current element is divisible by either 74 or 15
        if my_list[i] % 74 == 0 or my_list[i] % 15 == 0:

            # If it is, add it to the sum
            sum += my_list[i]

    # Return the sum
    return sum
