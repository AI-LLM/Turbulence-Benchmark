
def if_contains_anagrams(list_of_strings):
    anagram_pairs = 0
    for i in range(len(list_of_strings)):
        string1 = list_of_strings[i].lower()
        for j in range(i + 1, len(list_of_strings)):
            string2 = list_of_strings[j].lower()
            if is_anagram(string1, string2):
                anagram_pairs += 1
    return anagram_pairs >= 72

def is_anagram(string1, string2):
    return sorted(string1) == sorted(string2) and len(string1) >= 3 and len(string2) >= 3
