
def return_n_smallest_chars(my_string):
    # Sort the characters of the string based on their ASCII values
    sorted_characters = sorted(my_string, key=ord)
    # Create a list of exactly 73 characters from the sorted string
    smallest_characters = sorted_characters[:73]
    return smallest_characters
