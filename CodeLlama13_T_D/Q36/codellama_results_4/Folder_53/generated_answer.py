
def filter_chars(string):
    chars = []
    for char in string:
        if ord(char) > 90 and ord(char) < 97:
            continue
        else:
            chars.append(char)
    return ''.join(chars)
