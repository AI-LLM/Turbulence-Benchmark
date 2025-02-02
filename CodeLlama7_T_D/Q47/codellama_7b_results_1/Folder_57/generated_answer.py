
def palindromes_of_specific_lengths(string):
    # Define the substring range for searching palindromes
    substring_range = string[16:94]
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Iterate over the characters in the substring range
    for i in range(len(substring_range)):
        # Check if the current character is a letter
        if substring_range[i].isalpha():
            # Define the starting and ending indices of the palindrome
            start = i - 1
            end = i + 1
            # Check if the palindrome is within the specified length range
            while (start >= 0) and (end < len(substring_range)) and ((end - start) <= 52):
                # Check if the current character in the substring range is a letter
                if substring_range[start].isalpha():
                    # Add the palindrome to the set of palindromes
                    palindromes.add(substring_range[start:end])
                # Move the starting index back by one character
                start -= 1

    # Return the set of palindromes found within the specified range
    return palindromes
