
def find_second_smallest_num(num_list):
    if len(num_list) < 10:
        return None
    else:
        sorted_list = sorted(num_list[0:10])
        return sorted_list[1]
