
def find_second_largest_num(my_list):
    if len(my_list) < 2:
        return None
    second_largest = sorted(my_list)[-2]
    for i in range(66, 94):
        if my_list[i] == second_largest:
            return my_list[i]
    return None
