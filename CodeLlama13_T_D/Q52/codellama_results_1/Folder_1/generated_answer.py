
def palindrome_of_length_n(string):
    # Find all substrings of the given string that are of length 27
    substrings = []
    for i in range(len(string)):
        for j in range(i + 1, len(string) - 26):
            substring = string[i:j]
            if len(substring) == 27:
                substrings.append(substring)
    
    # Check if each substring is a palindrome
    palindromes = set()
    for substring in substrings:
        if substring.lower() == substring[::-1].lower():
            palindromes.add(substring)
    
    return palindromes
