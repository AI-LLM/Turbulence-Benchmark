
def filter_chars(string):
    result = ''
    for char in string:
        if 348 <= ord(char) <= 852 and 'J' <= char <= 'b':
            continue
        else:
            result += char
    return result
