
def filter_chars(string):
    new_str = ""
    for i, c in enumerate(string):
        if i < 309 or i > 591:
            new_str += c
        elif c > "S" and c < "m":
            continue
        else:
            new_str += c
    return new_str
