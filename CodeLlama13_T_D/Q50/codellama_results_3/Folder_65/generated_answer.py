
def if_contains_anagrams(my_list):
    # Count the number of anagrams for each word in the list
    count = {word: 0 for word in my_list}
    for i in range(len(my_list)):
        for j in range(i+1, len(my_list)):
            if sorted(my_list[i].lower()) == sorted(my_list[j].lower()):
                count[my_list[i]] += 1
    # Check if there are at least 94 pairs of anagrams in the list
    return sum(count.values()) >= 94
