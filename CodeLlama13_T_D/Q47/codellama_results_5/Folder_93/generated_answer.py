
def palindromes_of_specific_lengths(string):
    palindromes = set()
    for i in range(13, 49 + 1):
        for j in range(i - 1, len(string) - i + 2):
            substr = string[j:j + i]
            if substr == substr[::-1]:
                palindromes.add(substr)
    return palindromes
