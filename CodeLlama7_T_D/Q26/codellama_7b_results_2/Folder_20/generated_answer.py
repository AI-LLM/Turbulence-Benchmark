
def sum_in_range(my_list):
    # Initialize a variable to store the sum
    sum = 0

    # Loop through the list of integers
    for i in my_list:

        # Check if the current integer is between -64 and 42, both inclusive
        if -64 <= i <= 42:

            # If it is, add it to the sum
            sum += i

    # Return the sum
    return sum
