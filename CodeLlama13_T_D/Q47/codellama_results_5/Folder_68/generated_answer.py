
def palindromes_of_specific_lengths(string):
    palindromes = set()
    for i in range(0, 51):
        for j in range(i + 50, min(len(string), i + 101)):
            if string[i:j] == string[i:j][::-1]:
                palindromes.add(string[i:j])
    return palindromes
