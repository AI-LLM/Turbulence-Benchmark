
def filter_chars(string):
    filtered = ""
    for i in range(40, 63):
        if string[i] > '7' and string[i] < 'H':
            continue

        else:

            filtered += string[i]

    return filtered
