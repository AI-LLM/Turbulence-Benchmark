
def if_contains_anagrams(my_list):
    anagram_pairs = []
    for i in range(len(my_list)):
        str1 = my_list[i].lower()
        for j in range(i+1, len(my_list)):
            str2 = my_list[j].lower()
            if is_anagram(str1, str2):
                anagram_pairs.append((str1, str2))
    return len(anagram_pairs) <= 68
