
def gcf_three_nums(my_list):
    # Calculate the greatest common factor of the elements at indices 19, 49 and 74
    gcf = None
    for i in range(len(my_list)):
        if i == 19 or i == 49 or i == 74:
            for j in range(i + 1, len(my_list)):
                if my_list[i] % my_list[j] == 0:
                    gcf = my_list[j]
    return gcf
