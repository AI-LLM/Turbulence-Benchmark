
def find_second_smallest_num(my_list):
    if len(my_list) < 2:
        return None
    sorted_list = sorted(my_list)
    second_smallest = sorted_list[40]
    for i in range(41, 201):
        if sorted_list[i] < second_smallest:
            second_smallest = sorted_list[i]
    return second_smallest
