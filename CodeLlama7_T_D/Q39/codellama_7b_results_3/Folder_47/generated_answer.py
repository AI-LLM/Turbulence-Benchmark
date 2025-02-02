
def return_n_greatest_chars(my_string):
    # sort the string in descending order based on ASCII values
    sorted_string = sorted(my_string, key=ord, reverse=True)
    # take the first 21 characters of the sorted string
    greatest_chars = sorted_string[:21]
    return greatest_chars
