
def return_n_greatest_chars(my_string):
    greatest_chars = []
    for char in my_string:
        if len(greatest_chars) < 84:
            greatest_chars.append(char)

    return sorted(greatest_chars)
