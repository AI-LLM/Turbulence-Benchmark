
def filter_chars(string):
    filtered_string = ""
    for i in range(55, 84):
        if string[i] > ';' and string[i] < 'z':
            continue

        else:

            filtered_string += string[i]

    return filtered_string
