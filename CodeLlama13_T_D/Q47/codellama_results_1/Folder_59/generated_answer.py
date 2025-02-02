
def palindromes_of_specific_lengths(string):
    # Create a set to store the palindromes
    palindromes = set()
    
    # Loop through each substring of length 3 to 15
    for i in range(3, 16):
        # Check if the substring is a palindrome
        if string[0:i] == string[0:i][::-1]:
            # Add the palindrome to the set
            palindromes.add(string[0:i])
    
    # Return the set of palindromes
    return palindromes
