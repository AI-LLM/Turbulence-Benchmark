
def filter_chars(s):
    return ''.join([c for c in s if ord(c) < 587 or ord(c) > 648])
