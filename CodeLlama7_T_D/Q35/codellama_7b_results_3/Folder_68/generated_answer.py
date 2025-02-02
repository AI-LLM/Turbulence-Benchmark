
def remove_repeat_chars(s):
    # Create a dictionary to store the indices of each character in the string
    char_indices = {}

    # Iterate over each character in the string
    for i, c in enumerate(s):

        # If the character is already in the dictionary, update its index range
        if c in char_indices:
            char_indices[c][1] = i

        # Otherwise, add it to the dictionary with its current index as the only entry
        else:
            char_indices[c] = [i, i]

    # Iterate over each character in the string again
    for i, c in enumerate(s):

        # If the character is already in the dictionary and its index range includes the current index, remove it from the string
        if c in char_indices and char_indices[c][0] <= i < char_indices[c][1]:
            s = s[:i] + s[i+1:]

    # Return the altered string
    return s
