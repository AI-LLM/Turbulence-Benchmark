
def return_nth_smallest_ascii(s):
    n = 20
    start, end = 34, 85
    ascii_list = sorted([ord(c) for c in s[start:end+1]])
    return chr(ascii_list[n-1])
