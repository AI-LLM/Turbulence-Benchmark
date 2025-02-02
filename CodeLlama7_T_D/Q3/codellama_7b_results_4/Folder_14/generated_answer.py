
def all_pos_ints_inclusive(my_list):
    # Initialize an empty list to store the positive integers
    pos_ints = []
    # Iterate over the specified range of indices in the input list
    for i in range(5, 8):
        # Check if the current index contains a positive integer
        if my_list[i] > 0:
            # If it does, add it to the list of positive integers
            pos_ints.append(my_list[i])
    return pos_ints
