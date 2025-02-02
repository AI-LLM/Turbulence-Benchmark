
def remove_repeat_chars(s):
    # Create a dictionary to keep track of the characters that occur more than once between indices 36 and 52
    char_count = {}
    
    # Iterate through the string and count the number of occurrences of each character in the given range
    for i in range(37, 53):
        if s[i] not in char_count:
            char_count[s[i]] = 1
        else:
            char_count[s[i]] += 1
    
    # Remove all occurrences of each character that occurs more than once between indices 36 and 52 from the string
    for c in char_count:
        if char_count[c] > 1:
            s = s.replace(c, '')
    
    return s
