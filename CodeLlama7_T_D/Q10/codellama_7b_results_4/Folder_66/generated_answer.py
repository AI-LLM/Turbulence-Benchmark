
def all_odd_ints_exclusive(my_list):
    # Get the length of the list
    list_length = len(my_list)
    # Initialize an empty list to store the odd integers
    odd_integers = []
    # Iterate through the list from index 13 to index 76 (exclusive)
    for i in range(13, 76):
        # Check if the current element is an odd integer
        if my_list[i] % 2 != 0:
            # If it is, add it to the list of odd integers
            odd_integers.append(my_list[i])
    # Return the list of all odd integers from index 13 to index 76 (exclusive)
    return odd_integers
