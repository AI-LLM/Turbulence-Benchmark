
def remove_repeat_chars(s):
    # create a list of tuples containing the index and character of each occurrence of a repeated character
    repeat_char_indices = [(i, c) for i, c in enumerate(s) if s[i:].count(c) > 1]
    # sort the list by index ascending
    repeat_char_indices.sort(key=lambda x: x[0])
    # iterate over the sorted list and remove all occurrences of each repeated character
    for i, c in repeat_char_indices:
        s = s[:i] + s[i+1:]
    return s
