
def palindromes_between_indices(string):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Loop through each character in the string
    for i in range(3, len(string) - 8 + 1):
        # Check if the substring starting at index i and ending at index i + 7 is a palindrome
        if string[i:i + 7] == string[i:i + 7][::-1]:
            # Add the palindrome to the set
            palindromes.add(string[i:i + 7])
    return palindromes
