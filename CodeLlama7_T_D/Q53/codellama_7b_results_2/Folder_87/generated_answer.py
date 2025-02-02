
def palindrome_of_length_at_least_n(input_string, n):
    # Create a set to store the palindromes
    palindromes = set()
    # Loop through each character in the input string
    for i in range(len(input_string)):
        # Check if the current character is a letter (a-zA-Z)
        if input_string[i].isalpha():
            # Create a variable to store the palindrome
            palindrome = ""
            # Loop through each character in the input string starting from the current index and going backwards
            for j in range(i, -1, -1):
                # Check if the current character is a letter (a-zA-Z)
                if input_string[j].isalpha():
                    # Add the current character to the palindrome
                    palindrome += input_string[j]
            # Check if the palindrome has a length greater than or equal to 18
            if len(palindrome) >= n:
                # Add the palindrome to the set of palindromes
                palindromes.add(palindrome)
    # Return the set of palindromes
    return palindromes
