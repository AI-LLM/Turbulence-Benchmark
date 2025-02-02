
def sum_even_ints_inclusive(my_list):
    # Initialize a variable to store the sum
    sum = 0

    # Iterate through the list from index 37 to 51 (both inclusive)
    for i in range(37, 52):

        # If the current element is even, add it to the sum
        if my_list[i] % 2 == 0:
            sum += my_list[i]

    # Return the sum of all even integers from index 37 to index 51 (both inclusive), or 0 if no even integers exist in the specified range
    return sum
