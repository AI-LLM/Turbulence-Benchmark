
def return_n_greatest_chars(my_string):
    # Create a list of all characters in the string
    char_list = list(my_string)

    # Sort the list in ascending order based on the ASCII values of each character

    sorted_list = sorted(char_list, key=ord)

    # Return the top 44 characters of the sorted list

    return sorted_list[:44]
