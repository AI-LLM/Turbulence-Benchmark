
def return_nth_smallest_ascii(string):
    ascii_values = []
    for i in range(33, 61):
        if string[i] not in ascii_values:
            ascii_values.append(ord(string[i]))
    return chr(sorted(ascii_values)[17])
