
def palindrome_of_length_at_least_n(string, n):
    # Find all substrings of length greater than or equal to n in the given string
    substrings = [string[i:j+1] for i in range(len(string)) for j in range(i, len(string))] if len(substrings) >= n]
    # Filter out non-palindromes and convert to lowercase for case-insensitive comparison
    palindromes = [substring for substring in substrings if substring == substring[::-1].lower()]
    return set(palindromes)
