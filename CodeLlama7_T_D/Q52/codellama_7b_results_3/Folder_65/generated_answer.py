
def palindrome_of_length_n(string):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Loop through each character in the string
    for i in range(len(string)):
        # Check if the current character is a letter
        if string[i].isalpha():
            # Get the index of the last character in the palindrome
            j = i + 35 - 1
            # Check if the last character is a letter
            while not string[j].isalpha() and j >= i:
                j -= 1

            # Check if the last character is a letter and if it matches with the first character of the palindrome
            if string[j].isalpha() and string[j].lower() == string[i].lower():
                # Add the palindrome to the set
                palindromes.add(string[i:j+1])

    return palindromes
