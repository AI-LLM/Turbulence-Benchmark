
def filter_chars(my_string):
    # Create a list of characters in the given string
    char_list = list(my_string)
    # Iterate through the list of characters and remove any that are outside the specified range
    for i in range(len(char_list)):
        if (ord(char_list[i]) < 348 or ord(char_list[i]) > 852) and (ord(char_list[i]) < ord('J') or ord(char_list[i]) > ord('b')):
            char_list.pop(i)
    # Return the altered string with the filtered characters removed
    return "".join(char_list)
