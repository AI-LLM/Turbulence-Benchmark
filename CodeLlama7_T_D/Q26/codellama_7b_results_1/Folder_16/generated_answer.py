
def sum_in_range(my_list):
    # Initialize the sum to 0
    sum = 0
    # Loop through the list of integers
    for i in my_list:
        # Check if the integer is between 142 and 407, both inclusive
        if 142 <= i <= 407:
            # Add the integer to the sum
            sum += i

    # Return the sum
    return sum
