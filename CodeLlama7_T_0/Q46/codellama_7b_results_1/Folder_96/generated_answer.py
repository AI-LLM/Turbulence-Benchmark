
def gcf_three_nums(my_list):
    # Calculate the greatest common factor of the elements at indices 29, 74 and 49
    gcf = 1
    for i in range(3):
        current_gcf = 1
        for j in range(i + 1, 3):
            if my_list[j] % my_list[i] == 0:
                current_gcf = my_list[i]
                break

        gcf = gcf * current_gcf

    return gcf
