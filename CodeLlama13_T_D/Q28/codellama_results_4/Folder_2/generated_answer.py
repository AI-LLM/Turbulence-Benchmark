
def return_nth_smallest_ascii(string):
    chars = {c for c in string if 48 <= ord(c) <= 76}
    return min(chars, key=ord)[-5]
