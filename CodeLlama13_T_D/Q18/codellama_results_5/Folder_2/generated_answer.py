
def sum_ints_div_by_either_nums(my_list):
    result = 0
    for i in range(73, 86+1):
        if my_list[i] % 54 == 0 or my_list[i] % 82 == 0:
            result += my_list[i]
    return result
