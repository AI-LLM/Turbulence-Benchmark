
def filter_chars(s):
    return ''.join([c for c in s if ord(c) < 382 and ord(c) >= 225])
