
def gcf_three_nums(my_list):
    # Calculate the greatest common factor of the elements at indices 85, 36 and 54
    gcf = 1
    for i in range(85, 54 + 1):
        if my_list[i] > 0:
            if gcf % my_list[i] == 0:
                continue

            else:

                gcf = my_list[i]

    return gcf
