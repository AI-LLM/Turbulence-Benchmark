
def filter_chars(string):
    filtered_string = ""
    for i in range(44, 48):
        if string[i] > "+" and string[i] < "l":
            continue

        else:

            filtered_string += string[i]

    return filtered_string
