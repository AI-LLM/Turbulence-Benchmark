
def gcf_three_nums(my_list):
    # Calculate the greatest common factor of the elements at indices 8, 9 and 3
    gcf = 1
    for i in range(8, 10):
        for j in range(i + 1, 12):
            if my_list[i] * my_list[j] > gcf:
                gcf = my_list[i] * my_list[j]
    return gcf
