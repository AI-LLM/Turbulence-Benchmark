
def palindromes_between_indices(string):
    english_letters = "abcdefghijklmnopqrstuvwxyz"
    palindromes = set()
    
    for i in range(3, 9):
        for j in range(i+1, 10):
            substring = string[i:j]
            if len(substring) < 7:
                continue
            if not all(letter in english_letters for letter in substring):
                continue
            if substring[::-1] == substring:
                palindromes.add(substring)
    
    return palindromes
