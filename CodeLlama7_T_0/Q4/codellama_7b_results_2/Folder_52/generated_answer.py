
def all_pos_ints_exclusive(my_list):
    # Initialize an empty list to store the positive integers
    pos_ints = []
    # Iterate through the list of integers
    for i in range(len(my_list)):
        # Check if the current integer is positive and greater than index 93
        if my_list[i] > 0 and i > 93:
            # Add the positive integer to the list of positive integers
            pos_ints.append(my_list[i])
    # Return the list of all positive integers from index 93 to index 94, both exclusive
    return pos_ints
