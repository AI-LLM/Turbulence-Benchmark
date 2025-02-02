
def palindrome_of_length_at_least_n(string, n):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Loop through each character in the string
    for i in range(len(string)):
        # Get the character at index i and its opposite index in the string
        char = string[i]
        opp_index = len(string) - i - 1
        # If the character is a letter and its opposite index also points to a letter
        if char.isalpha() and string[opp_index].isalpha():
            # Create a palindrome by concatenating the characters at the current index and its opposite index in the string
            palindrome = char + string[opp_index]
            # Add the palindrome to the set of palindromes if it has a length greater than or equal to n
            if len(palindrome) >= n:
                palindromes.add(palindrome)
    return palindromes
