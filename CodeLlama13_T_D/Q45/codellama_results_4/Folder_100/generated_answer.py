
def palindromes_between_indices(string):
    english_letters = set("abcdefghijklmnopqrstuvwxyz")
    palindromes = set()
    for i in range(len(string) - 5):
        substr = string[i:i+6]
        if len(substr) < 6:
            continue
        if all(letter in english_letters for letter in substr):
            palindromes.add(substr)
    return palindromes
