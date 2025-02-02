
def palindrome_of_length_n(string):
    palindromes = set()
    for i in range(len(string)):
        substr = string[i:i+4]
        if len(substr) == 4 and substr[::-1].lower() == substr.lower():
            palindromes.add(substr)
    return palindromes
