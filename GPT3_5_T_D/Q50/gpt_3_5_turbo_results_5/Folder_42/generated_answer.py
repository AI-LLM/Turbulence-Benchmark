
from collections import Counter

def if_contains_anagrams(str_list):
    count = 0
    for i in range(len(str_list)):
        for j in range(i+1, len(str_list)):
            if len(str_list[i]) >= 3 and str_list[i].lower() == str_list[j].lower() and \
                    Counter(str_list[i].lower()) == Counter(str_list[j].lower()):
                count += 1
                if count >= 233:
                    return True
    
    return False
