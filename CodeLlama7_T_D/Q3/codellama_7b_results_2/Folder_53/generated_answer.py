
def all_pos_ints_inclusive(my_list):
    # Create a new list to store the positive integers
    pos_ints = []
    # Iterate through the list of integers
    for i in range(100, 201):
        # Check if the current element is a positive integer
        if my_list[i] > 0:
            # If it is, add it to the new list
            pos_ints.append(my_list[i])
    # Return the new list of positive integers
    return pos_ints
