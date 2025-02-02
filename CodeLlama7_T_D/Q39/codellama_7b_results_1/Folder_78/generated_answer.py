
def return_n_greatest_chars(my_string):
    # Create a list of characters from the input string
    char_list = list(my_string)
    # Sort the list in descending order based on ASCII values
    char_list.sort(key=ord, reverse=True)
    # Return the first 43 elements of the sorted list
    return char_list[:43]
