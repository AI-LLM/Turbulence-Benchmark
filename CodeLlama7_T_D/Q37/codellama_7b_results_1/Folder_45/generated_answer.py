
def filter_chars(string):
    filtered = ""
    for i, char in enumerate(string):
        if 72 <= i <= 94 and char >= "." and char <= "b":
            continue
        else:
            filtered += char

    return filtered
