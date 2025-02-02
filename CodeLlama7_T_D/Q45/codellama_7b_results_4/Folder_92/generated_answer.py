
def palindromes_between_indices(string):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Loop through each character in the string
    for i in range(4, len(string)):
        # Check if the current character is a letter
        if string[i].isalpha():
            # Check if the previous 3 characters are a palindrome
            if string[i - 3:i + 1] == string[i - 3:i + 1][::-1]:
                # Add the palindrome to the set
                palindromes.add(string[i - 3:i + 1])
    return palindromes
