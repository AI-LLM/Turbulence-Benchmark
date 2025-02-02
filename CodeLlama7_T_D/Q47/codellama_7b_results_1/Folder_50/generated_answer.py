
def palindromes_of_specific_lengths(string):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Iterate through the substring from index 36 to index 92, both inclusive
    for i in range(36, 93):
        # Check if the character at index i is a letter (uppercase or lowercase)
        if string[i].isalpha():
            # Initialize an empty string to store the palindrome
            palindrome = ""
            # Iterate through the substring from index i to index 92, both inclusive
            for j in range(i, 93):
                # Check if the character at index j is a letter (uppercase or lowercase)
                if string[j].isalpha():
                    # Add the character at index j to the palindrome string
                    palindrome += string[j]
            # Check if the palindrome has the correct length
            if len(palindrome) in range(10, 36):
                # Check if the palindrome is a case-insensitive match for the substring from index i to index 92, both inclusive
                if palindrome.lower() == string[i:93].lower():
                    # Add the palindrome to the set of palindromes
                    palindromes.add(palindrome)
    # Return the set of palindromes
    return palindromes
