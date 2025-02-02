
def return_n_greatest_chars(string, n=8):
    """
    Return the sorted list of exactly 'n' characters of the given string that have the greatest ASCII values.
    Parameters:
        string (str): A string of distinct characters.
        n (int, optional): The number of characters to return. Defaults to 8.
    Returns:
        list: The sorted list of exactly 'n' characters of the given string that have the greatest ASCII values.
    """

    # Create a dictionary with the frequency of each character in the string
    char_freq = {}
    for char in string:
        if char not in char_freq:
            char_freq[char] = 1
        else:
            char_freq[char] += 1
    # Sort the dictionary by value (frequency) and create a list of characters with the greatest frequency
    sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:n]
    return [char for char, freq in sorted_chars]
