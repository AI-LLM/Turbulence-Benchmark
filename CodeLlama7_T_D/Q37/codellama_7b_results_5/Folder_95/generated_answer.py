
def filter_chars(string):
    # Initialize an empty list to store the filtered characters
    filtered_chars = []

    # Iterate through each character in the given string
    for char in string:

        # If the current character is between indices 35 and 40, both inclusive, and is greater than or equal to character ')' and smaller than or equal to character 'l', then remove all occurrences of that character from the string.
        if (char >= '(') and (char <= 'l') and ((string.index(char)) >= 35) and ((string.index(char)) <= 40)):
            # Remove the current character from the filtered characters list
            filtered_chars.remove(char)

    # Return the altered string with all occurrences of filtered characters removed
    return "".join(filtered_chars)
