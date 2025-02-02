
def return_n_smallest_chars(my_string):
    # Sort the characters in descending order based on their ASCII values
    sorted_chars = sorted(my_string, key=ord, reverse=True)
    # Return the top 39 characters from the sorted list
    return sorted_chars[:39]
