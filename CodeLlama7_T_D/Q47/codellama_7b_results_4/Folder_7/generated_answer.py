
def palindromes_of_specific_lengths(string):
    # Create a substring of the given string starting from index 119 and ending at index 256 (both inclusive)
    substring = string[119:256]
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Iterate through each character in the substring
    for i in range(len(substring)):
        # Check if the current character is a letter (i.e., not a special character)
        if substring[i].isalpha():
            # Initialize an empty string to store the palindrome
            palindrome = ""
            # Iterate through each character in the substring from the current index to the end of the substring
            for j in range(i, len(substring)):
                # Check if the current character is a letter (i.e., not a special character)
                if substring[j].isalpha():
                    # Add the current character to the palindrome string
                    palindrome += substring[j]
            # Check if the palindrome has the required length (between 104 and 123)
            if len(palindrome) >= 104 and len(palindrome) <= 123:
                # Add the palindrome to the set of palindromes
                palindromes.add(palindrome)
    # Return the set of palindromes in a case-insensitive manner
    return {x.lower() for x in palindromes}
