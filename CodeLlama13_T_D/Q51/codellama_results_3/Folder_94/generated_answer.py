
def if_contains_anagrams(list_of_strings):
    anagrams = set()
    for s in list_of_strings:
        lower_s = s.lower()
        sorted_s = "".join(sorted(lower_s))
        if sorted_s not in anagrams:
            anagrams.add(sorted_s)
        else:
            return False
    return len(anagrams) <= 181
