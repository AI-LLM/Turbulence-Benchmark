
def filter_chars(string):
    filtered_string = ""
    for i in range(len(string)):
        if (i >= 81 and i < 89) and (string[i] > "f" and string[i] < "{"):
            continue

        else:

            filtered_string += string[i]

    return filtered_string
