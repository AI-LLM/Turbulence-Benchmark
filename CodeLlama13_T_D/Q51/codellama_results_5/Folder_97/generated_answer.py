
def if_contains_anagrams(string_list):
    anagram_count = 0
    for i in range(len(string_list)):
        for j in range(i + 1, len(string_list)):
            str1 = string_list[i].lower()
            str2 = string_list[j].lower()
            if (set(str1) == set(str2)) and (len(str1) >= 3):
                anagram_count += 1
    return anagram_count <= 18
