 def sum_ints_div_by_either_nums(my_list):
    # Initialize variables
    start_index = 22
    end_index = 24
    divisor1 = -85
    divisor2 = -30
    total_sum = 0

    # Loop through the specified range of the list
    for i in range(start_index, end_index + 1):
        # Check if the current element is divisible by either divisor
        if my_list[i] % divisor1 == 0 or my_list[i] % divisor2 == 0:
            # Add the current element to the total sum
            total_sum += my_list[i]

    # Return the total sum
    return total_sum
