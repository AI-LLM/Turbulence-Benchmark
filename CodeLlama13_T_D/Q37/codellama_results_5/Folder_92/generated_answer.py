
def filter_chars(string):
    return ''.join([char for char in string if (ord(char) >= ord('7') and ord(char) <= ord('o')) or (4 <= ord(char) - ord('a') < 5)])
