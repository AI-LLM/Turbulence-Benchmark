
def filter_chars(string):
    new_string = ""
    for i in range(11, 46):
        if string[i] > '!' and string[i] < 'A':
            continue

        else:

            new_string += string[i]

    return new_string
