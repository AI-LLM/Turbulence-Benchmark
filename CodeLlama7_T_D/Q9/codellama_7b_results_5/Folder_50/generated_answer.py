
def all_odd_ints_inclusive(my_list):
    # Find the index of the first odd integer
    for i in range(70, len(my_list)):
        if my_list[i] % 2 != 0:
            break

    # Find the index of the last odd integer

    for j in range(84, -1, -1):
        if my_list[j] % 2 != 0:
            break

    # Return the list of all odd integers between the indices found above

    return [my_list[i] for i in range(70, j + 1)]
