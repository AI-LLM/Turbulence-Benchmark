
def filter_chars(string):
    # Remove all characters between 48 and 75 (exclusive) from the string
    filtered_string = ''.join([char for char in string if not (48 < ord(char) < 76)])
    return filtered_string
