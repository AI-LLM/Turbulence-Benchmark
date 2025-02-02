
def if_contains_anagrams(strings):
    anagrams = []
    for string in strings:
        lowercase_string = string.lower()
        if len(lowercase_string) >= 3 and not lowercase_string in anagrams:
            anagrams.append(lowercase_string)
    return len(anagrams) <= 70
