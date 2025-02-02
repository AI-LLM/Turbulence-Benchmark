 def return_n_greatest_chars(string, n=47):
    """
    Returns the sorted list of exactly n characters of the given string that have the greatest ASCII values.
    Args:
        string (str): The input string of distinct characters.
        n (int, optional): The number of characters to return. Defaults to 47.
    Returns:
        list[str]: The sorted list of exactly n characters of the given string that have the greatest ASCII values.
    """

    # Create a dictionary to store the counts of each character in the input string
    char_counts = {}
    for char in string:
        if char not in char_counts:
            char_counts[char] = 1
        else:
            char_counts[char] += 1

    # Sort the dictionary by value (descending) and then by key (ascending)
    sorted_counts = sorted(char_counts.items(), key=lambda x: (-x[1], x[0])))

    # Extract the n greatest characters from the sorted dictionary
    greatest_chars = []
    for char, count in sorted_counts[:n]:
        greatest_chars.append(char)

    # Return the sorted list of exactly n greatest characters of the given string
    return sorted(greatest_chars)
