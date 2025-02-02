
def palindromes_of_specific_lengths(string):
    palindromes = set()
    for i in range(18, 87+1):
        for j in range(i, min(len(string), i+60)):
            if string[i:j] == string[i:j][::-1]:
                palindromes.add(string[i:j])
    return palindromes
