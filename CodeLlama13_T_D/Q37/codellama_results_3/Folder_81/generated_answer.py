
def filter_chars(string):
    return ''.join([char for char in string if (19 <= ord(char) <= 92 and ord('8') <= ord(char) <= ord('m'))])
