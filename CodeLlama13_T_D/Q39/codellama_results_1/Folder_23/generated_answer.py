
def return_n_greatest_chars(string):
    chars = sorted(set(string), key=lambda x: ord(x))
    return chars[-46:]
