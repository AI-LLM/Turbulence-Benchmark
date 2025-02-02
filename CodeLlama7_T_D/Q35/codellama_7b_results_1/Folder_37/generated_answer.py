
def remove_repeat_chars(s):
    # create a set of unique characters in the string
    unique_chars = set(s)
    # loop through each character in the string
    for char in s:
        # if the character is not in the set of unique characters, skip it
        if char not in unique_chars:
            continue

        # if the character occurs more than once between indices 2 and 7 (exclusive), remove all occurrences of that character from the string
        if s.count(char) > 1 and len(s[2:7]) != 0:
            s = s.replace(char, '', s.count(char))
    return s
