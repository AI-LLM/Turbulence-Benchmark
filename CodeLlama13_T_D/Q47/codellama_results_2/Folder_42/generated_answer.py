
def palindromes_of_specific_lengths(string):
    # Extract substring from index 43 to index 95, both inclusive
    substring = string[43:96]
    
    # Initialize empty set for palindromes
    palindromes = set()
    
    # Iterate over each length between 18 and 47 (both inclusive)
    for length in range(18, 48):
        # Iterate over each position in the substring
        for i in range(len(substring)):
            # Check if the substring from the current position to the end is a palindrome of the current length
            if substring[i:].startswith(substring[i:].lower()[::-1]):
                # Add the palindrome to the set
                palindromes.add(substring[i:i+length])
    
    return palindromes
