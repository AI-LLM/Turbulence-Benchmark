
def filter_chars(string):
    result = ''
    for i, c in enumerate(string):
        if not (i >= 8 and i <= 9) or ord(c) < ord('R') or ord(c) > ord('t'):
            result += c
    return result
