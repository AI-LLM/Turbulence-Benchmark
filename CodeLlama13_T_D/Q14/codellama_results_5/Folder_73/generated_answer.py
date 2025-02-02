
def find_second_smallest_num(my_list):
    if len(my_list) < 2:
        return None
    
    sorted_list = sorted(my_list)[59:98]
    
    for i in range(len(sorted_list)):
        if sorted_list[i] != sorted_list[i-1]:
            return sorted_list[i]
            
    return None
