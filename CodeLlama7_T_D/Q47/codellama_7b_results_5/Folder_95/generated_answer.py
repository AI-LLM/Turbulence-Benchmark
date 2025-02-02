 def palindromes_of_specific_lengths(string):
    # Define the range of indices for which we want to find palindromes
    start_index = 12
    end_index = 56
    # Define the minimum and maximum lengths of palindromes that we want to find
    min_len = 20
    max_len = 21
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Iterate over the characters in the substring
    for i in range(start_index, end_index + 1):
        # Check if the character at index i is a letter
        if string[i].isalpha():
            # Iterate over the possible lengths of palindromes that can be formed using the character at index i
            for j in range(min_len, max_len + 1):
                # Check if the length of the palindrome is within the specified range
                if j >= min_len and j <= max_len:
                    # Check if the palindrome formed using the character at index i and the length j is a palindrome
                    if string[i - j + 1:i + 1] == string[i - j + 1:i + 1][::-1]:
                        # Add the palindrome to the set of palindromes
                        palindromes.add(string[i - j + 1:i + 1])
    # Return the set of palindromes that meet the specified requirements
    return palindromes
