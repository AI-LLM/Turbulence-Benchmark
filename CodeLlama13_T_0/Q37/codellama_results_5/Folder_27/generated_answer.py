
def filter_chars(string):
    result = ""
    for i, char in enumerate(string):
        if 43 <= ord(char) <= 80 and '=' <= char <= 'E':
            continue
        result += char
    return result
