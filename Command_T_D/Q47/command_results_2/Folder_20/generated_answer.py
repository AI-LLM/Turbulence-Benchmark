import re
def palindromes_of_specific_lengths(s):
    return set(re.findall(r"<[a-z]+>((?:[a-z]+){36,42})<[a-z]+>", s[20:-5]))
