
def if_contains_anagrams(lst):
    counter = 0
    for i in range(len(lst)):
        for j in range(i+1,len(lst)):
            if len(lst[i]) >= 3 and sorted(lst[i].lower()) == sorted(lst[j].lower()):
                counter += 1
                if counter >= 3:
                    return True
    return False
