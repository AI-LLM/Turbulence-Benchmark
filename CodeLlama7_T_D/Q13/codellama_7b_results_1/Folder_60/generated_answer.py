
def find_second_largest_num(my_list):
    # Find the largest element in the list
    largest = max(my_list)

    # Create a new list that contains all elements from index 75 to index 85, both inclusive
    sublist = my_list[75:86]

    # Find the second largest element in the sublist
    second_largest = max(sublist)

    # If the second largest element is not equal to the largest element, return it
    if second_largest != largest:
        return second_largest

    # Otherwise, return 'None'
    else:
        return None
