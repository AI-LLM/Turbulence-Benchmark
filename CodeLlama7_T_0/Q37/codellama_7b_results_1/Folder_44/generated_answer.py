
def filter_chars(string):
    filtered_string = ""
    for i in range(21, 43 + 1):
        if string[i] >= "+" and string[i] <= "8":
            continue

        else:

            filtered_string += string[i]

    return filtered_string
