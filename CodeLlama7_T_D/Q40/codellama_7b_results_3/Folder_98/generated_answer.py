
def return_n_smallest_chars(string):
    # sort the string in descending order based on ASCII values
    sorted_string = sorted(string, key=ord, reverse=True)
    # create a list of exactly 68 characters from the sorted string
    smallest_chars = sorted_string[:68]
    return smallest_chars
