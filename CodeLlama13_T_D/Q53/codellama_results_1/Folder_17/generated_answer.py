
def palindrome_of_length_at_least_n(string):
    palindromes = set()
    for i in range(len(string) - 36 + 1):
        substring = string[i:i+36]
        if len(substring) < 36 or not all(c.isalpha() for c in substring):
            continue
        if substring == substring[::-1]:
            palindromes.add(substring)
    return palindromes
