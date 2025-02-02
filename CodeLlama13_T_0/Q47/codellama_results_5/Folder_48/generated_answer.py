
def palindromes_of_specific_lengths(string):
    palindromes = set()
    for i in range(155, 284):
        for j in range(i+103, i+121):
            if string[i:j] == string[i:j][::-1]:
                palindromes.add(string[i:j])
    return palindromes
