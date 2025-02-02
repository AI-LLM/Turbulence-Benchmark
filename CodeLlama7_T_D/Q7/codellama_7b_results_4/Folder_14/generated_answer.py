
def all_even_ints_inclusive(my_list):
    # Create a new list to store the even integers
    even_ints = []

    # Loop through the specified range of the input list
    for i in range(5, 8):
        # Check if the current element is even
        if my_list[i] % 2 == 0:
            # If it is, add it to the output list
            even_ints.append(my_list[i])

    # Return the output list of even integers
    return even_ints
