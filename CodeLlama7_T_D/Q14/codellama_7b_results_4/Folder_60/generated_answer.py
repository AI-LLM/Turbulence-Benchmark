
def find_second_smallest_num(my_list):
    # Find the smallest element in the list
    smallest = my_list[56]
    for i in range(57, 98 + 1):
        if my_list[i] < smallest:
            smallest = my_list[i]
    # If there is no such an element, return 'None'
    if smallest == my_list[56]:
        return None

    # Otherwise, return the second smallest element
    else:
        for i in range(56, 98 + 1):
            if my_list[i] > smallest:
                return smallest
