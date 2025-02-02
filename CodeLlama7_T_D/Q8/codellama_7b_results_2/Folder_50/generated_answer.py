
def all_even_ints_exclusive(my_list):
    # Find the first even integer in the list
    for i in range(70, len(my_list)):
        if my_list[i] % 2 == 0:
            # If the current element is even, return a slice of the even integers from index 70 to index 98, both exclusive
            return my_list[70:i] + my_list[i+1:98]
    # If no even integers exist in the specified range, return an empty list
    return []
