
def find_second_largest_num(my_list):
    if len(my_list) <= 1:
        return None
    
    second_largest = my_list[56]
    for i in range(57, 67):
        if my_list[i] > second_largest:
            second_largest = my_list[i]
            
    return second_largest
