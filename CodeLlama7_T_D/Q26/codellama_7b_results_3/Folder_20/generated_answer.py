
def sum_in_range(my_list):
    # Initialize variables
    sum = 0
    range_start = -64
    range_end = 42

    # Iterate over the list and check if each element is between the specified range
    for i in my_list:
        if range_start <= i <= range_end:
            sum += i

    return sum
