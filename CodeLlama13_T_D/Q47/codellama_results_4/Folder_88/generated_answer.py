
def palindromes_of_specific_lengths(string):
    palindromes = set()
    for i in range(11, 87 + 1):
        for j in range(i, min(len(string), 87) + 1):
            substr = string[i:j]
            if len(substr) >= 4 and len(substr) <= 5:
                rev_substr = substr[::-1]
                if substr == rev_substr:
                    palindromes.add(substr)
    return palindromes
