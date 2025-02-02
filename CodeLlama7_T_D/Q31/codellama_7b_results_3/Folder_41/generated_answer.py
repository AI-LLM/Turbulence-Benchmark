
def if_perfect_num(my_list):
    # Check that the list is non-empty and has at least 84 elements
    if len(my_list) < 84:
        return False

    # Check if the integer at index 84 is a perfect number

    for i in range(2, int(my_list[84] ** 0.5) + 1):
        if my_list[84] % i == 0:
            j = 2
            while i * j <= my_list[84]:
                if my_list[84] % (i * j) == 0:
                    return False

                j += 1

    return True
