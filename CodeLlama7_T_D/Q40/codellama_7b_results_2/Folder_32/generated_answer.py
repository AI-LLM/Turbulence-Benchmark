
def return_n_smallest_chars(my_string):
    # sort the string in ascending order based on ASCII value
    sorted_list = sorted(my_string, key=ord)
    # create a list of exactly 8 characters from the sorted list
    smallest_chars = sorted_list[:8]
    return smallest_chars
