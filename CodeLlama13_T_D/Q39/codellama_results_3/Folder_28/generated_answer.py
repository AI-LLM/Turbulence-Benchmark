
def return_n_greatest_chars(string):
    sorted_list = sorted(string, key=lambda x: ord(x))
    return sorted_list[len(sorted_list) - 3:]
