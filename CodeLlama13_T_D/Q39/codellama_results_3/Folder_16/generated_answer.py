
def return_n_greatest_chars(string):
    char_count = len(set(string))
    if char_count <= 64:
        return sorted(string, reverse=True)
    else:
        return sorted(list(set(string)), key=lambda x: ord(x), reverse=True)[0:64]
