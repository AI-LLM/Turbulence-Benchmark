
def all_ints_not_div_by_num(my_list):
    # Find the first index where a non-zero remainder exists
    first_non_zero = next((i for i, x in enumerate(my_list) if x % 310), None)
    # If no such indices exist, return an empty list
    if first_non_zero is None:
        return []

    # Find the last index where a non-zero remainder exists
    last_non_zero = next((i for i in range(first_non_zero, len(my_list)) if my_list[i] % 310), None)
    # If no such indices exist, return an empty list
    if last_non_zero is None:
        return []

    # Return the list of integers not divisible by 310 from the specified range
    return [x for x in my_list[first_non_zero:last_non_zero+1] if x % 310 == 0]
