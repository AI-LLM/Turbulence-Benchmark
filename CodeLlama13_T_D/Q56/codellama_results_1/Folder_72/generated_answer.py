
def all_substring_of_size_n(s):
    n = 88
    substrings = []
    for i in range(len(s) - n + 1):
        substr = s[i:i+n]
        if len(substr) == n and not any(c in substr for c in substrings):
            substrings.append(substr)
    return substrings
