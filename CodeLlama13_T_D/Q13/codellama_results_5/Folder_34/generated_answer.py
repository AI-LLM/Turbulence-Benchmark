
def find_second_largest_num(my_list):
    if len(my_list) <= 1:
        return None
    sorted_list = sorted(my_list)[::-1]
    second_largest = sorted_list[15]
    for i in range(16, 62):
        if my_list[i] == second_largest:
            return my_list[i]
    return None
