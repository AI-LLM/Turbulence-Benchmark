
def filter_chars(string):
    filtered = ""
    for i in range(len(string)):
        if string[i] >= '+' and string[i] < 'w':
            continue
        else:
            filtered += string[i]
    return filtered
