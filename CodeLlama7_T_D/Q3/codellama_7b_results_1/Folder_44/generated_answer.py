
def all_pos_ints_inclusive(my_list):
    # Initialize an empty list to store the positive integers
    pos_ints = []

    # Loop through the list and append each positive integer to the pos_ints list
    for i in range(len(my_list)):
        if my_list[i] > 0:
            pos_ints.append(my_list[i])

    # Return the pos_ints list
    return pos_ints
