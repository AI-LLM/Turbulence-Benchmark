
def filter_chars(string):
    result = ""
    for char in string:
        if ord(char) < 61 or ord(char) > 70 or ord(char) < 48 or ord(char) > 64:
            result += char
    return result
