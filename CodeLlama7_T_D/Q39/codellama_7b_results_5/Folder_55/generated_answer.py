
def return_n_greatest_chars(input_str, n=75):
    # convert input string to a list of characters
    char_list = list(input_str)
    # sort the list in descending order based on ASCII values
    sorted_list = sorted(char_list, key=ord, reverse=True)
    # return the top n characters from the sorted list
    return sorted_list[:n]
