import re
def palindromes_between_indices(s):
    return set(re.findall(r'(?i)(?=([a-z]){3}[a-z]){4}', s[3:-1]))
