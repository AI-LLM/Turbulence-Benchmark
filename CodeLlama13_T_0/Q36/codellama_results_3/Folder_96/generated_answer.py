
def filter_chars(string):
    result = ""
    for i, char in enumerate(string):
        if i < 38 or i > 81:
            result += char
        elif char > "." and char < "^":
            continue
        else:
            result += char
    return result
