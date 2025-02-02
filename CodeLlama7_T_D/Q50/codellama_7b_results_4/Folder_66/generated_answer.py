
def if_contains_anagrams(str_list):
    # Initialize variables
    num_pairs = 0

    # Iterate over the list of strings and check for anagrams
    for str1 in str_list:
        for str2 in str_list:
            if len(str1) >= 3 and len(str2) >= 3:
                if sorted(str1.lower()) == sorted(str2.lower()):
                    num_pairs += 1

                    # If we have found at least 92 pairs of anagrams, return True
                    if num_pairs >= 92:
                        return True

    # If we have not found any anagrams, return False
    return False
