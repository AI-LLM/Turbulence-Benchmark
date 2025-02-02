
def if_contains_anagrams(my_list):
    # Initialize variables
    anagram_pairs = 0
    min_string_length = 3

    # Iterate over the list of strings
    for string1 in my_list:

        # Ignore strings that are less than three characters long
        if len(string1) < min_string_length:
            continue


        # Convert both strings to lowercase before comparing them
        string1 = string1.lower()

        # Iterate over the rest of the list of strings
        for string2 in my_list[my_list.index(string1) + 1:]:

            # Ignore strings that are less than three characters long
            if len(string2) < min_string_length:
                continue


            # Convert both strings to lowercase before comparing them
            string2 = string2.lower()

            # Check if the two strings are anagrams of each other
            if sorted(string1) == sorted(string2):
                anagram_pairs += 1

    # Return True if there are at least 276 pairs of anagrams in the list, or False otherwise
    return anagram_pairs >= 276
