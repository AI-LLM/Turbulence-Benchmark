
import re

def palindromes_of_specific_lengths(string):
    substring = string[70:141]
    palindromes = set()
    
    for length in range(3, 61):
        for i in range(len(substring) - length + 1):
            sub = substring[i:i+length]
            if sub.lower() == sub.lower()[::-1] and re.match("^[a-zA-Z]+$", sub):
                palindromes.add(sub)
    
    return palindromes
