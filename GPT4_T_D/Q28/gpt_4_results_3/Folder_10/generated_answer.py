
def return_nth_smallest_ascii(s):
    s = s[38:83]
    s = sorted(s, key=ord)
    return s[19]
