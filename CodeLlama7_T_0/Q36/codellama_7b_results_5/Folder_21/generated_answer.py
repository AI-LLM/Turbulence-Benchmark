
def filter_chars(string):
    filtered_string = ""
    for i in range(219, 403):
        if string[i] > '*' and string[i] < '7':
            continue

        else:

            filtered_string += string[i]

    return filtered_string
