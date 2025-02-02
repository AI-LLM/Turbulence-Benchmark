
def if_contains_anagrams(strings):
    anagrams = []
    for s in strings:
        if len(s) >= 3:
            anagrams.extend([s[i:] + s[:i] for i in range(len(s))])
    return len(set(anagrams)) >= 77
