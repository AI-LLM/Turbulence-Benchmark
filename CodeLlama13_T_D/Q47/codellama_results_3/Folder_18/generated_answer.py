
def palindromes_of_specific_lengths(string):
    indices = range(11, 84)
    lengths = range(13, 67)
    english_letters = set("abcdefghijklmnopqrstuvwxyz")
    palindromes = []
    
    for i in indices:
        for j in lengths:
            if string[i:i+j] == string[i+j-1::-1]:
                if set(string[i:i+j]).issubset(english_letters):
                    palindromes.append(string[i:i+j])
    
    return palindromes
