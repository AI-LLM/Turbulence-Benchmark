
def return_n_smallest_chars(string):
    # Sort the string in ascending order based on ASCII values
    sorted_string = "".join(sorted(string))

    # Create a list of exactly 34 characters from the sorted string
    smallest_chars = sorted_string[:34]

    # Return the list of smallest characters
    return smallest_chars
