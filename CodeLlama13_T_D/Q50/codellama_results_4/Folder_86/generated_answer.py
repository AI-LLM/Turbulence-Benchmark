
def if_contains_anagrams(strings):
    pairs = 0
    for i in range(len(strings)):
        for j in range(i+1, len(strings)):
            if "".join(sorted(strings[i].lower())) == "".join(sorted(strings[j].lower())):
                pairs += 1
    return pairs >= 115
