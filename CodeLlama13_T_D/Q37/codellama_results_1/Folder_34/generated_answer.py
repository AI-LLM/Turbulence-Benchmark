
def filter_chars(string):
    chars = []
    for i in range(54, 63):
        if string[i].isdigit() and ')' <= string[i] <= '9':
            chars.append(string[i])
    for char in chars:
        string = string.replace(char, '')
    return string
