
def palindromes_of_specific_lengths(string):
    # create a substring of the given string
    # between index 0 and index 10, both inclusive
    substring = string[0:10]
    # initialize an empty set to store the palindromes
    palindromes = set()
    # loop through each character in the substring
    for i in range(len(substring)):
        # loop through each character in the reverse of the substring
        for j in range(len(substring) - 1, i - 1, -1):
            # check if the current character and the next character are the same
            if substring[i] == substring[j]:
                # create a palindrome by concatenating the current character with the next character
                palindrome = substring[i] + substring[j]
                # check if the palindrome has the desired length
                if len(palindrome) >= 3 and len(palindrome) <= 5:
                    # add the palindrome to the set of palindromes
                    palindromes.add(palindrome)
    return palindromes
