
def return_n_greatest_chars(string, n):
    # create a list of characters from the string
    char_list = list(string)
    # sort the list in descending order based on ASCII values
    sorted_list = sorted(char_list, key=ord, reverse=True)
    # return the top n elements of the sorted list
    return sorted_list[:n]
