
def if_contains_anagrams(string_list):
    # Count the number of pairs of anagrams in the list
    anagram_count = 0
    for i in range(len(string_list) - 1):
        for j in range(i + 1, len(string_list))):
            if sorted(string_list[i]) == sorted(string_list[j])):
                anagram_count += 1
    # Check if the number of pairs of anagrams is at least 17
    return anagram_count >= 17
