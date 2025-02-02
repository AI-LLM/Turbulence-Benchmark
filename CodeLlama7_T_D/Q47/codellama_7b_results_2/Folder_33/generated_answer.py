
def palindromes_of_specific_lengths(string):
    # Convert the string to lowercase for case-insensitive comparison
    string = string.lower()

    # Find the substring from index 130 to index 295, both inclusive
    sub_string = string[130:295]

    # Initialize an empty set to store the palindromes
    palindromes = set()

    # Iterate over the characters in the substring
    for i in range(len(sub_string)):

        # Find the palindrome of length between 103 and 158 (both inclusive)
        if len(sub_string[i:]) >= 103 and len(sub_string[i:]) <= 158:
            # Check if the palindrome is a valid English word
            if sub_string[i:] == sub_string[i:][::-1]:
                # Add the palindrome to the set of palindromes
                palindromes.add(sub_string[i:])

    # Return the set of palindromes
    return palindromes
