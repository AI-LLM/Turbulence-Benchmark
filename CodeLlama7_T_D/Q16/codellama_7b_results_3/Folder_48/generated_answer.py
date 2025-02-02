
def sum_even_ints_inclusive(my_list):
    # Initialize a variable to store the sum
    total = 0

    # Iterate through the list from index 533 to index 605, inclusive
    for i in range(533, 606):

        # If the current element is even, add it to the sum
        if my_list[i] % 2 == 0:
            total += my_list[i]

    # Return the sum of all even integers from index 533 to index 605, inclusive. If no even integers exist in the specified range, return 0.
    return total
