
def sum_odd_ints_inclusive(my_list):
    # Initialize a variable to store the sum
    total = 0

    # Iterate through the list, starting from index 30 and ending at index 87 (inclusive)
    for i in range(30, 88):

        # Check if the current element is an odd integer
        if my_list[i] % 2 != 0:

            # If it is, add it to the total sum
            total += my_list[i]

    # Return the total sum
    return total
