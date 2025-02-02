
import string

def if_contains_anagrams(my_list):
    # Create a set of all the lowercase letters
    all_letters = set(string.ascii_lowercase)
    # Initialize variables to keep track of anagram pairs
    pair_count = 0
    for i in range(len(my_list)):
        word1 = my_list[i]
        for j in range(i + 1, len(my_list))):
            word2 = my_list[j]
            # Check if the words are anagrams of each other
            if sorted(word1) == sorted(word2):
                pair_count += 1

    # Return True if there are at most 47 pairs of anagrams in the given list; otherwise, return False
    return pair_count <= 47
