
def return_n_smallest_chars(string):
    char_list = []
    for c in string:
        if len(char_list) < 7:
            char_list.append(c)
        elif ord(c) < ord(char_list[-1]):
            char_list.pop()
            char_list.append(c)
    return char_list
