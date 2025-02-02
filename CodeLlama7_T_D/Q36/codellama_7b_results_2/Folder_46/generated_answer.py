
def filter_chars(my_string):
    # Create a list of characters in the given string
    char_list = list(my_string)
    # Iterate through each character in the list
    for i in range(len(char_list)):
        # Check if the current character is between 19 and 69 (exclusive)
        if 19 < ord(char_list[i]) < 69:
            # If so, remove all occurrences of that character within the string
            char_list = [char for char in char_list if char != char_list[i]]
    # Return the altered string
    return "".join(char_list)
