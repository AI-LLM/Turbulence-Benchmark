
def return_nth_smallest_ascii(string):
    # Create a list of the characters in the string, sorted by their ASCII value
    chars = sorted(list(set(string)))
    
    # Return the character with the 11th smallest ASCII value that occurs between indices 20 and 80
    return chars[11]
