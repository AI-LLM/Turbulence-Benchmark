
def filter_chars(string):
    result = ""
    for i, char in enumerate(string):
        if i < 27 or i > 85:
            result += char
        elif char >= "!" and char <= "B":
            continue
        else:
            result += char
    return result
