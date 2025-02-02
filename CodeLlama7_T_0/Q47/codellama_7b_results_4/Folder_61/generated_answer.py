
def palindromes_of_specific_lengths(string):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Loop through each character in the string
    for i in range(len(string)):
        # Get the substring starting from index i to index 30 (both inclusive)
        substring = string[i:31]
        # Loop through each palindrome of length between 20 and 30 (both inclusive) in the substring
        for j in range(20, 31):
            # Check if the substring is a palindrome
            if substring == substring[::-1]:
                # Add the palindrome to the set of palindromes
                palindromes.add(substring)
    # Return the set of palindromes in a case-insensitive manner
    return {palindrome.lower() for palindrome in palindromes}
