
def filter_chars(s):
    new_str = ""
    for i, c in enumerate(s[51:76]):
        if ord(c) > ord('5') and ord(c) < ord('f'):
            new_str += s[:i+1] + s[i+2:]
    return new_str
