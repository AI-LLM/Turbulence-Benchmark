
def find_second_smallest_num(my_list):
    if len(my_list) < 2:
        return None
    sorted_list = sorted(my_list)
    second_smallest = sorted_list[41]
    for i in range(42, 69):
        if my_list[i] == second_smallest:
            continue
        else:
            return None
    return second_smallest
