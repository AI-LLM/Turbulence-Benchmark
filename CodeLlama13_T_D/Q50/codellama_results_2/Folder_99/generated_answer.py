
def if_contains_anagrams(strings):
    anagrams = set()
    for string in strings:
        key = "".join(sorted(string.lower()))
        if key not in anagrams:
            anagrams.add(key)
    return len(anagrams) >= 314
