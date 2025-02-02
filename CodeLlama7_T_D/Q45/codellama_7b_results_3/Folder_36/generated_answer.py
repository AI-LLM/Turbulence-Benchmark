 def palindromes_between_indices(string):
    # Initialize an empty set to store the palindromes
    palindromes = set()
    # Iterate over each letter in the given string that occurs between index 5 to index 8, both inclusive
    for i in range(5, 9):
        # Create a list of all English letters that occur before and after the current letter
        left_letters = string[:i]
        right_letters = string[i+1:]
        # Iterate over each possible arrangement of the English letters in the given string that occur between index 5 to index 8, both inclusive
        for j in range(26):
            # Create a list of all English letters that have been rearranged according to the current iteration
            letters = [left_letters[k] for k in range(len(left_letters))) + [string[i]] + [right_letters[k] for k in range(len(right_letters)))]
            # Form a palindrome by reversing the list of letters and comparing it to the original list of letters
            if letters == letters[::-1]:
                # Add the palindrome to the set of palindromes
                palindromes.add("".join(letters))
    return palindromes
