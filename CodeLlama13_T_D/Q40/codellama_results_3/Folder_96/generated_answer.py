
def return_n_smallest_chars(string):
    char_list = []
    for i in range(20):
        smallest = min(string, key=ord)
        char_list.append(smallest)
        string = string.replace(smallest, "")
    return sorted(char_list, reverse=True)
