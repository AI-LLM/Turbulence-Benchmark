
def return_n_greatest_chars(string):
    char_list = sorted(set(string), key=lambda x: ord(x), reverse=True)[:44]
    return char_list
