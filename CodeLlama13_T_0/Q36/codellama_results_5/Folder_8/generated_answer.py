
def filter_chars(string):
    for i in range(82, 93):
        if string[i] > '!' and string[i] < '*':
            string = string.replace(string[i], '')
    return string
