
def palindromes_of_specific_lengths(string):
    palindromes = set()
    for i in range(75, 96):
        for j in range(i + 7, min(i + 10, len(string))):
            if string[i:j] == string[i:j][::-1]:
                palindromes.add(string[i:j])
    return palindromes
