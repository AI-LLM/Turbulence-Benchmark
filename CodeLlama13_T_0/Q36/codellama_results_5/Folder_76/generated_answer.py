
def filter_chars(string):
    new_str = ""
    for i, char in enumerate(string):
        if i < 669 or i > 888:
            new_str += char
        elif char > "B" and char < "Y":
            new_str += char
    return new_str
