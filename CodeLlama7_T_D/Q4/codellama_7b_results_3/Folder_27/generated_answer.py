
def all_pos_ints_exclusive(my_list):
    # Initialize an empty list to store the positive integers
    pos_ints = []

    # Iterate through the list of integers from index 28 to index 49, both exclusive
    for i in range(28, 50):

        # Check if the current integer is positive
        if my_list[i] > 0:

            # Add the positive integer to the list of positive integers
            pos_ints.append(my_list[i])

    # Return the list of all positive integers from index 28 to index 49, both exclusive
    return pos_ints
