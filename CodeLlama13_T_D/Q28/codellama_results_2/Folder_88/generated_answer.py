
def return_nth_smallest_ascii(string):
    ascii_list = [ord(char) for char in string[38:64]]
    nth_smallest = sorted(ascii_list)[14]
    return chr(nth_smallest)
