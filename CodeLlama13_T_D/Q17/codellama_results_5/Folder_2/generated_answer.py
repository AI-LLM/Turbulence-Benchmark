
def all_ints_div_by_both_two_nums(my_list):
    result = []
    for i in range(73, 86 + 1):
        if my_list[i] % 54 == 0 and my_list[i] % 82 == 0:
            result.append(my_list[i])
    return result
