
def return_n_smallest_chars(string):
    # create a list of tuples containing (character, ASCII value) for each character in the string
    char_list = [(char, ord(char)) for char in string]
    # sort the list in descending order based on the ASCII value
    char_list.sort(key=lambda x: x[1], reverse=True)
    # return the top 33 characters from the sorted list
    return char_list[:33]
