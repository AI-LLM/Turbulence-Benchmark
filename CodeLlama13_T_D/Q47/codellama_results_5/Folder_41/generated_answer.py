
def palindromes_of_specific_lengths(string):
    palindromes = set()
    for i in range(1, 7 + 1):
        for j in range(i + 2, 7 + 1):
            substr = string[i:j]
            if len(substr) >= 3 and len(substr) <= 4:
                if substr == substr[::-1]:
                    palindromes.add(substr)
    return palindromes
