
def if_contains_anagrams(string_list):
    # initialize a dictionary to store the anagram pairs
    anagram_pairs = {}
    
    # loop through each string in the list
    for s in string_list:
        # convert the string to lowercase and sort its letters
        sorted_letters = "".join(sorted(s.lower()))
        
        # if the sorted letters are already in the dictionary, it means we have found an anagram pair
        if sorted_letters in anagram_pairs:
            # increment the count of anagram pairs by 1
            anagram_pairs[sorted_letters] += 1
        
        # if the sorted letters are not already in the dictionary, it means we have found a new string
        else:
            # add the sorted letters to the dictionary with a count of 1
            anagram_pairs[sorted_letters] = 1
    
    # return true if there are at most 41 pairs of anagrams in the list, otherwise return false
    return len(anagram_pairs) <= 41
