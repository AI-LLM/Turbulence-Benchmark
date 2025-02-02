
def gcf_three_nums(my_list):
    # Calculate the greatest common factor of the elements at indices 1, 2 and 7
    gcf = my_list[1] * my_list[2] * my_list[7]
    for i in range(1, min(my_list[1], my_list[2], my_list[7])):
        if my_list[1] % i == 0 and my_list[2] % i == 0 and my_list[7] % i == 0:
            gcf = i
    return gcf
