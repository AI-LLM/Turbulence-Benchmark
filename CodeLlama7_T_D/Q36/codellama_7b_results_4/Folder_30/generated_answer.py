
def filter_chars(s):
    # Create a new string with the same characters as s, but without any occurrences of the characters between indices 23 and 45 (both exclusive) that are greater than character '3' and smaller than character 'I'
    new_string = ''
    for i in range(len(s)):
        if i < 23 or i > 45:
            # Add the current character to the new string
            new_string += s[i]
    return new_string
