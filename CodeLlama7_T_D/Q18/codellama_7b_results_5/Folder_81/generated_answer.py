
def sum_ints_div_by_either_nums(my_list):
    # Initialize variables
    sum = 0
    # Iterate through the list from index 10 to index 100, both inclusive
    for i in range(10, 101):
        # Check if the current element is divisible by either 10 or 100
        if my_list[i] % 10 == 0 or my_list[i] % 100 == 0:
            # If it is, add it to the sum
            sum += my_list[i]
    # Return the sum
    return sum
