
def filter_chars(s):
    return ''.join([c for c in s if not (54 <= ord(c) <= 83 and 'j' <= c <= 'v')])
