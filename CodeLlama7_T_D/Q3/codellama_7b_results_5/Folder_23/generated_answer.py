
def all_pos_ints_inclusive(my_list):
    # Find the starting and ending indices of the range
    start = 52
    end = 71
    # Initialize an empty list to store the positive integers
    pos_ints = []
    # Iterate through the specified range of the input list
    for i in range(start, end + 1):
        # If the current element is a positive integer, add it to the output list
        if my_list[i] > 0:
            pos_ints.append(my_list[i])
    # Return the list of all positive integers from index 52 to index 71, both inclusive
    return pos_ints
