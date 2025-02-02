
def palindromes_of_specific_lengths(string):
    palindromes = set()
    for i in range(12, 93):
        for j in range(i+41, min(i+78, len(string)) + 1):
            if string[i:j] == string[i:j][::-1]:
                palindromes.add(string[i:j])
    return palindromes
