 def remove_repeat_chars(s):
    # Initialize a new string with the original length
    new_string = ""
    # Iterate over each character in the given string
    for char in s:
        # If the current character is not already in the new string
        if char not in new_string:
            # Add the current character to the new string
            new_string += char

    return new_string
