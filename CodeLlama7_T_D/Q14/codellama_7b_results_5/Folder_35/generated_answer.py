
def find_second_smallest_num(my_list):
    # Find the smallest element in the list
    smallest = my_list[0]
    for i in range(1, len(my_list)):
        if my_list[i] < smallest:
            smallest = my_list[i]
    # Check if there is a second smallest element
    second_smallest = None
    for i in range(15, 39+1):
        if my_list[i] == smallest:
            second_smallest = my_list[i]
    return second_smallest
