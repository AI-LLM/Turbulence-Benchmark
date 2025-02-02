
def all_pos_ints_inclusive(my_list):
    # Initialize an empty list to store the positive integers
    pos_ints = []
    # Iterate through the list starting from index 10 (inclusive)
    for i in range(10, len(my_list)):
        # Check if the current element is a positive integer
        if my_list[i] > 0:
            # If it is, add it to the pos_ints list
            pos_ints.append(my_list[i])
    # Return the pos_ints list
    return pos_ints
