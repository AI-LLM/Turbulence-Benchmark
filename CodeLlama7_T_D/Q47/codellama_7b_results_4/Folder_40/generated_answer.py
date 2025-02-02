
def palindromes_of_specific_lengths(string):
    # Define the substring and the palindrome lengths of interest
    substring = string[:6]
    palindrome_lengths = range(3, 6)
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Iterate over each palindrome length of interest
    for length in palindrome_lengths:
        # Use a generator expression to find all occurrences of palindromes of the current length in the substring
        palindromes_at_current_length = (word for word in substring.split() if len(word) == length and word == word[::-1])
        # Add each palindrome found to the set of palindromes
        palindromes.update(palindromes_at_current_length)
    # Return the set of palindromes found in the substring
    return palindromes
