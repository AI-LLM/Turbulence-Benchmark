
def all_substring_of_size_n(s):
    substrings = []
    for i in range(len(s) - 59):
        substring = s[i:i+60]
        if len(set(substring)) == 60 and not any(c in substrings for c in substring):
            substrings.append(substring)
    return substrings
