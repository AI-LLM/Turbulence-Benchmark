
def find_second_smallest_num(my_list):
    if len(my_list) <= 1:
        return None

    sorted_list = sorted(my_list)[50:201]

    if len(sorted_list) < 3:
        return None

    return sorted_list[2]
