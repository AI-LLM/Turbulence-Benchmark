
def return_nth_smallest_ascii(string):
    # create a list of all the characters in the string
    char_list = list(string)

    # sort the list of characters by their ASCII values
    sorted_char_list = sorted(char_list, key=ord)

    # return the character with the 18th smallest ASCII value
    return sorted_char_list[17]
