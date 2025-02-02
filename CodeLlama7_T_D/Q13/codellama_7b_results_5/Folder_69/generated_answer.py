
def find_second_largest_num(my_list):
    # Check if the input is a list
    if not isinstance(my_list, list):
        return None

    # Check if the list is empty

    if len(my_list) == 0:
        return None

    # Find the second largest element

    second_largest = None

    for i in range(32, 35 + 1):
        if my_list[i] > second_largest or second_largest is None:
            second_largest = my_list[i]

    # Return the second largest element

    return second_largest
