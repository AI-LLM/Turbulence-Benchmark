
def filter_chars(string):
    result = ""
    for char in string:
        if ord(char) > 109 and ord(char) < 123:
            continue
        else:
            result += char
    return result
