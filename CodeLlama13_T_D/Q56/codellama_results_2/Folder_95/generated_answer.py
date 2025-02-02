
def all_substring_of_size_n(s):
    result = []
    for i in range(len(s) - 89 + 1):
        substring = s[i:i+89]
        if len(set(substring)) == 89 and all(substring.count(c) == 1 for c in substring):
            result.append(substring)
    return result
