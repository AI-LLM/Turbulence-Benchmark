
def return_n_greatest_chars(string):
    char_dict = {}
    for c in string:
        if c not in char_dict:
            char_dict[c] = ord(c)
    
    sorted_char_list = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)[:69]
    return [c for c, _ in sorted_char_list]
