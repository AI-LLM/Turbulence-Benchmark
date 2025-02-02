
def palindromes_of_specific_lengths(string):
    # Define the substring range for palindromes
    substr_start = 18
    substr_end = 65
    # Initialize an empty set to store palindromes
    palindromes = set()
    # Iterate through each character in the substring
    for i in range(substr_start, substr_end + 1):
        # Check if the current character is a letter (case-insensitive)
        if string[i].isalpha():
            # Define the palindrome with the current character at its center
            palindrome = string[i - 23: i + 24]
            # Check if the palindrome has the required length and consists of only English letters
            if len(palindrome) >= 23 and len(palindrome) <= 36 and all(char.isalpha() for char in palindrome)):
                # Add the palindrome to the set of palindromes
                palindromes.add(palindrome)
    return palindromes
