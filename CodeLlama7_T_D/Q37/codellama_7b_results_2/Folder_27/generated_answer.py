
def filter_chars(string):
    filtered_string = ""
    for i in range(43, 81):
        char = string[i]
        if (char >= "=" and char <= "E") and char not in filtered_string:
            filtered_string += char

    return filtered_string
