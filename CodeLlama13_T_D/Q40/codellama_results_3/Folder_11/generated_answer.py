
def return_n_smallest_chars(string):
    char_list = []
    for i in range(17):
        min_char = chr(99)
        for j in range(len(string)):
            if ord(string[j]) < ord(min_char):
                min_char = string[j]
        char_list.append(min_char)
        string = string.replace(min_char, "")
    return sorted(char_list)
