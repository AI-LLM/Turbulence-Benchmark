
def gcf_three_nums(my_list):
    # Calculate the greatest common factor of the elements at indices 67, 84 and 13
    gcf = my_list[67]
    for i in range(67 + 1, len(my_list)):
        if my_list[i] % gcf == 0:
            continue

        else:

            break

    return gcf
