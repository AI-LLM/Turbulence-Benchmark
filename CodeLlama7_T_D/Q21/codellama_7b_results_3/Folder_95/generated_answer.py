
def all_ints_div_by_num(my_list):
    result = []
    for i in range(75, 86):
        if my_list[i] % -40 == 0:
            result.append(my_list[i])
    return result
