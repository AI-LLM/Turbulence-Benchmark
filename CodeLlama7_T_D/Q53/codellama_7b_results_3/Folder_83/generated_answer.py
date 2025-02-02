
def palindrome_of_length_at_least_n(string, n):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Loop through each character in the string
    for i in range(len(string)):
        # Check if the character is a letter (a-z or A-Z)
        if string[i].isalpha():
            # Convert the character to lowercase
            char = string[i].lower()
            # Loop through each character in the remainder of the string
            for j in range(i + 1, len(string)):
                # Check if the character is a letter (a-z or A-Z)
                if string[j].isalpha():
                    # Convert the character to lowercase
                    other_char = string[j].lower()
                    # Check if the two characters are the same and are part of a palindrome of length greater than or equal to n
                    if char == other_char and len(char + other_char) >= n:
                        # Add the palindrome to the set of palindromes
                        palindromes.add(char + other_char)
    # Return the set of palindromes
    return palindromes
