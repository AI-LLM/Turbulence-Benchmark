
def palindromes_of_specific_lengths(string):
    substrings = [string[i:j] for i in range(3, 300) for j in range(i + 1, 300)]
    palindromes = set()
    for substring in substrings:
        if len(substring) >= 30 and len(substring) <= 300:
            if substring == substring[::-1]:
                palindromes.add(substring)
    return palindromes
