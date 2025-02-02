
def palindromes_between_indices(string):
    english_letters = set("abcdefghijklmnopqrstuvwxyz")
    palindromes = set()
    for i in range(1, 9):
        for j in range(i+6, len(string)+1):
            substr = string[i:j]
            if len(substr) >= 7 and all(c in english_letters for c in substr):
                palindromes.add(substr)
    return palindromes
