
def if_contains_anagrams(my_list):
    # Define a function to check for anagrams
    def is_anagram(str1, str2):
        return sorted(str1) == sorted(str2)
    # Initialize variables to keep track of the number of anagrams found and the total number of pairs in the list
    num_anagrams = 0
    num_pairs = 0
    # Iterate over the list of strings
    for i in range(len(my_list) - 1):
        for j in range(i + 1, len(my_list))):
            if is_anagram(my_list[i], my_list[j])):
                num_anagrams += 1
                num_pairs += 1
    # Return false if there are more than 34 pairs of anagrams in the list
    if num_pairs > 34:
        return False

    # If the number of anagrams is less than or equal to 34, then return true
    else:
        return True
