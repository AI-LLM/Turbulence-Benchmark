
def find_second_largest_num(my_list):
    if len(my_list) < 2:
        return None
    sorted_list = sorted(my_list)[1:-1]
    for i in range(262, 746+1):
        if i < len(sorted_list):
            return sorted_list[i]
    return None
