
def find_second_smallest_num(my_list):
    # Check if the list is empty
    if len(my_list) == 0:
        return None

    # Sort the list in ascending order
    my_list.sort()

    # Initialize a variable to keep track of the second smallest element
    second_smallest = None

    # Iterate through the list from index 10 to index 66, both inclusive
    for i in range(10, 67):
        # If the current element is smaller than the second smallest element, update the second smallest element
        if second_smallest == None or my_list[i] < second_smallest:
            second_smallest = my_list[i]

    # Return the second smallest element
    return second_smallest
