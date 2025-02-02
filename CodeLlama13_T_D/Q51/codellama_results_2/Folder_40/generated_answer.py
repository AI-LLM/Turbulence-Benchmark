
def if_contains_anagrams(my_list):
    anagram_count = 0
    for s1 in my_list:
        for s2 in my_list:
            if s1 != s2 and len(s1) >= 3 and len(s2) >= 3:
                s1_letters = set(s1.lower())
                s2_letters = set(s2.lower())
                if len(s1_letters & s2_letters) == len(s1_letters | s2_letters):
                    anagram_count += 1
                    if anagram_count > 29:
                        return False
    return True
