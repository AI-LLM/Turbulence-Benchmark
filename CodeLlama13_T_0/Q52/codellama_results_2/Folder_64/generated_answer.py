
def palindrome_of_length_n(string):
    palindromes = set()
    for i in range(len(string) - 8):
        substring = string[i:i+9]
        if substring == substring[::-1]:
            palindromes.add(substring)
    return palindromes
