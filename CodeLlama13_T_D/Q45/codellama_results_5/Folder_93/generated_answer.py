
def palindromes_between_indices(string):
    indices = range(2, 8)
    letters = "abcdefghijklmnopqrstuvwxyz"
    palindromes = set()
    
    for i in indices:
        for j in indices:
            if i <= j:
                substring = string[i:j]
                if len(substring) >= 6 and substring == substring[::-1]:
                    palindromes.add(substring)
    
    return palindromes
