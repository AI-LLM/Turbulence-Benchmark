
def find_second_largest_num(my_list):
    # Find the largest element in the list
    largest = max(my_list)

    # Create a new list with all elements that are not the largest
    non_largest = [x for x in my_list if x != largest]

    # If there is only one element left, return None
    if len(non_largest) == 1:
        return None

    # Otherwise, find the second largest element
    second_largest = max(non_largest)

    # Return the second largest element
    return second_largest
