
def palindromes_between_indices(string):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Loop through each character in the string
    for i in range(2, len(string)):
        # Check if the character is a letter
        if string[i].isalpha():
            # Create a copy of the string without the first and last characters
            substring = string[i + 1:len(string) - 1]
            # Loop through each character in the substring
            for j in range(len(substring)):
                # Check if the character is a letter
                if substring[j].isalpha():
                    # Create a palindrome by reversing the order of the letters
                    palindrome = ''.join(reversed(substring[j + 1:len(substring) - 1]])))
                    # Check if the palindrome is at least 4 characters long
                    if len(palindrome) >= 4:
                        # Add the palindrome to the set of palindromes
                        palindromes.add(palindrome)
    # Return the set of palindromes
    return palindromes
