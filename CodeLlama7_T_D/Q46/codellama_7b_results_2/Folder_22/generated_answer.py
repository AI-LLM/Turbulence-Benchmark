
def gcf_three_nums(my_list):
    # Calculate the greatest common factor of the elements at indices 27, 85 and 51
    gcf = 1
    for i in range(27, 86, 51):
        if my_list[i] % gcf == 0:
            continue

        else:

            while my_list[i] % gcf != 0:
                gcf += 1

    return gcf
