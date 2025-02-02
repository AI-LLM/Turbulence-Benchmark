
def remove_repeat_chars(s):
    # Create a set of unique characters in the string
    unique_chars = set(s)
    # Initialize an empty string to store the altered string
    altered_string = ""

    # Iterate over each character in the original string
    for char in s:
        # If the current character is not in the set of unique characters, add it to the altered string
        if char not in unique_chars:
            altered_string += char

        # If the current character is in the set of unique characters, check if it occurs more than once between indices 56 and 88
        elif s.count(char) > 1 and (s.index(char) >= 56 and s.index(char) <= 88)):
            # If it does, remove all occurrences of the character from the string
            altered_string += char

    return altered_string
