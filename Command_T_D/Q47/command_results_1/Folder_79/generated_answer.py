import re
def palindromes_of_specific_lengths(s):
    return set(map(str.lower, re.findall(r'(?i)([a-z]+)(?=([a-z]+\1)', s[1:-1])))
