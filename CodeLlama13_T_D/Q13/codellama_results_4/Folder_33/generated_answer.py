
def find_second_largest_num(my_list):
    second_largest = None
    largest = my_list[667]
    for i in range(668, 775):
        if my_list[i] > largest:
            largest = my_list[i]
        elif my_list[i] == largest and second_largest is None:
            second_largest = largest
    return second_largest
