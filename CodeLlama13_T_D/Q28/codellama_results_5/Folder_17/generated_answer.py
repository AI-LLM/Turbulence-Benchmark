
def return_nth_smallest_ascii(string):
    ascii_values = [ord(c) for c in string[4:33]]
    sorted_ascii_values = sorted(ascii_values)
    nth_smallest = sorted_ascii_values[3]
    return chr(nth_smallest)
