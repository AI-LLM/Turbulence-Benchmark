
def palindromes_between_indices(string):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Loop through each character in the string
    for i in range(len(string)):
        # Get the character at index i
        char = string[i]
        # If the character is a letter (not a space or punctuation)
        if char.isalpha():
            # Loop through each character in the string starting from index i + 1
            for j in range(i + 1, len(string)):
                # Get the character at index j
                other_char = string[j]
                # If the character is a letter (not a space or punctuation) and it is not the same as the first character
                if other_char.isalpha() and other_char != char:
                    # Check if the palindrome can be formed by reversing the order of the two characters
                    if char + other_char == other_char + char:
                        # If the palindrome is at least 7 characters long, add it to the set of palindromes
                        if len(char + other_char) >= 7:
                            palindromes.add(char + other_char)
    # Return the set of palindromes
    return palindromes
