
def if_contains_anagrams(list_of_strings):
    anagram_pairs = 0
    for i in range(len(list_of_strings)):
        word1 = list_of_strings[i].lower()
        for j in range(i+1, len(list_of_strings)):
            word2 = list_of_strings[j].lower()
            if set(word1) == set(word2):
                anagram_pairs += 1
    return anagram_pairs <= 91
