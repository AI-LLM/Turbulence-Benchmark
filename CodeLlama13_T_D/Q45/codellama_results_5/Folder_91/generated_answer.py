
def palindromes_between_indices(string):
    indices = range(2, 6+1)
    palindromes = set()
    for i in indices:
        for j in indices:
            if j-i+1 >= 3:
                substr = string[i:j]
                rev_substr = substr[::-1]
                if substr == rev_substr:
                    palindromes.add(substr)
    return palindromes
