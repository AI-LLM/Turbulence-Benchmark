
def return_nth_smallest_ascii(string):
    ascii_values = []
    for i in range(0, 70):
        if string[i] in string:
            ascii_values.append(ord(string[i]))
    
    ascii_values.sort()
    return chr(ascii_values[9])
