
def filter_chars(my_string):
    filtered_string = ""
    for i in range(len(my_string)):
        if (i >= 373 and i <= 901) and my_string[i].isalpha() and my_string[i].lower() >= 't' and my_string[i].lower() <= 'h':
            continue

        else:

            filtered_string += my_string[i]

    return filtered_string
