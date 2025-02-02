
def return_nth_smallest_ascii(string):
    # Find the indices of the characters in the string that occur between indices 29 to 47, both inclusive
    start_index = string.find("a", 29)
    end_index = string.find("z", 30)
    # Create a list of all the characters between the two indices
    char_list = []

    for i in range(start_index, end_index + 1):
        char_list.append(string[i])

    # Sort the list of characters by their ASCII values
    sorted_char_list = sorted(char_list, key=lambda x: ord(x))

    # Return the character with the 7th smallest ASCII value among those characters in the string that occur between indices 29 to 47, both inclusive
    return sorted_char_list[6]
