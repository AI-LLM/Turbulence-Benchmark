
def return_n_smallest_chars(string):
    # sort the string in ascending order based on ASCII value
    sorted_string = sorted(string, key=ord)
    # extract the first 68 characters from the sorted string
    smallest_characters = sorted_string[:68]
    return smallest_characters
