
def return_nth_smallest_ascii(string):
    ascii_values = []
    for char in string:
        if 47 <= ord(char) <= 70:
            ascii_values.append(ord(char))
    
    sorted_values = sorted(ascii_values)
    return chr(sorted_values[13])
