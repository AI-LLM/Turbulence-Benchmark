
def filter_chars(my_string):
    # Create a list of all characters in the given string
    char_list = list(my_string)

    # Iterate over each character in the list, checking if it falls within the specified range and is a valid character for the given string
    for i in range(len(char_list)):
        if 32 <= ord(char_list[i]) <= 46 and '0' <= char_list[i] <= 'k':
            # If the current character falls within the specified range, remove it from the list of characters
            del char_list[i]
    # Return the altered string with all occurrences of invalid characters removed
    return "".join(char_list)
