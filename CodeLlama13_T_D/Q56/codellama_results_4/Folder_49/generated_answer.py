
def all_substring_of_size_n(s):
    substrings = []
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            if j-i == 73:
                substring = s[i:j]
                if all(c not in substring for c in substring):
                    substrings.append(substring)
    return substrings
