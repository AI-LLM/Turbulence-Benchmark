
def if_contains_anagrams(list_of_strings):
    anagram_pairs = 0
    for i in range(len(list_of_strings)):
        for j in range(i+1, len(list_of_strings)):
            if sorted([x.lower() for x in list_of_strings[i]]) == sorted([y.lower() for y in list_of_strings[j]]):
                anagram_pairs += 1
    return anagram_pairs <= 4
