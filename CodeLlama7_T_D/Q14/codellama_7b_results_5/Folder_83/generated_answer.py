
def find_second_smallest_num(my_list):
    if len(my_list) < 2:
        return None

    smallest = my_list[0]
    second_smallest = None

    for i in range(1, len(my_list)):
        if my_list[i] < smallest:
            second_smallest = smallest
            smallest = my_list[i]

        elif my_list[i] < second_smallest or second_smallest is None:
            second_smallest = my_list[i]

    if second_smallest is not None:
        return second_smallest

    else:

        return None
