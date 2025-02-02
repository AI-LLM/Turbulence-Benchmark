
def palindromes_of_specific_lengths(string):
    # Define the range of lengths to consider
    lengths = range(40, 73)
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Iterate over each length in the range
    for length in lengths:
        # Get the substring starting from index 15 and ending at index 89 (both inclusive)
        substring = string[15:89]
        # Find all palindromes of the current length in the substring
        palindromes_in_substring = [word for word in substring.split() if len(word) == length and word == word[::-1]]
        # Add the found palindromes to the set of palindromes
        palindromes |= set(palindromes_in_substring)
    # Return the set of palindromes in a case-insensitive manner
    return {word.lower() for word in palindromes}
