
def all_ints_div_by_num(my_list):
    divisible_ints = []
    for i in range(62, 71):
        if my_list[i] % 58 == 0:
            divisible_ints.append(my_list[i])
    return divisible_ints
