
def filter_chars(string):
    result = ''
    for i, char in enumerate(string):
        if 1 <= ord(char) < 6 and char != ',':
            result += char
    return result
