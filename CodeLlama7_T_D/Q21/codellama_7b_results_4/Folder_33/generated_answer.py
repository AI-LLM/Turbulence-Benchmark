
def all_ints_div_by_num(my_list):
    # Find the first index where the remainder of division by -215 is non-zero
    first_non_zero = next((i for i, x in enumerate(my_list) if x % -215), None)
    # If no such indices exist, return an empty list
    if first_non_zero is None:
        return []

    # Find the last index where the remainder of division by -215 is non-zero
    last_non_zero = next((i for i in range(first_non_zero, len(my_list)) if my_list[i] % -215), None)
    # If no such indices exist, return an empty list
    if last_non_zero is None:
        return []

    # Return the list of all integers divisible by -215 from index 319 to index 612, both inclusive
    return my_list[first_non_zero:last_non_zero+1]
