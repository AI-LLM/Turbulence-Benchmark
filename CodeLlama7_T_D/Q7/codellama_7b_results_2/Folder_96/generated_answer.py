
def all_even_ints_inclusive(my_list):
    # Create a new list to store the even integers
    even_ints = []
    # Loop through the specified range of indices
    for i in range(50, 201):
        # If the current index contains an even integer, append it to the 'even_ints' list
        if my_list[i] % 2 == 0:
            even_ints.append(my_list[i])
    # Return the 'even_ints' list
    return even_ints
