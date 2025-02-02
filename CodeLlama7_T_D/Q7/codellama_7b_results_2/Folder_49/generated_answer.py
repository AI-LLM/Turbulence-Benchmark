
def all_even_ints_inclusive(my_list):
    # create a new list to store the even integers
    even_ints = []

    # loop through the list of integers
    for i in range(80, 201):
        # check if the current integer is even
        if my_list[i] % 2 == 0:
            # if it is even, add it to the new list
            even_ints.append(my_list[i])

    # return the list of all even integers from index 80 to index 200, both inclusive
    return even_ints
