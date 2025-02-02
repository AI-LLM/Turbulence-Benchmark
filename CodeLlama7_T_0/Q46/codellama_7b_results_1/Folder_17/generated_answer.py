
def gcf_three_nums(my_list):
    # Calculate the greatest common factor of the elements at indices 20, 51 and 62
    gcf = my_list[20]
    for i in range(len(my_list)):
        if my_list[i] != 0:
            while my_list[i] % gcf == 0:
                gcf = my_list[i]
    return gcf
