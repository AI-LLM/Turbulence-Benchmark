
def all_ints_div_by_num(my_list):
    result = []
    for i in range(20, 201, 4):
        if my_list[i] % 4 == 0:
            result.append(my_list[i])
    return result
