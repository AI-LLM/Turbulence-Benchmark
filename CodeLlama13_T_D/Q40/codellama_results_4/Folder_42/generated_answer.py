
def return_n_smallest_chars(string):
    char_list = []
    for i in range(59):
        char_list.append([i, string[i]])
    sorted_char_list = sorted(char_list, key=lambda x:x[1])
    return [x[1] for x in sorted_char_list]
