
def if_contains_anagrams(list_of_strings):
    # Initialize a counter for anagram pairs
    pair_counter = 0

    # Iterate over the list of strings
    for string1 in list_of_strings:

        # Convert the current string to lowercase
        string1_lower = string1.lower()

        # Iterate over the remaining strings in the list
        for string2 in list_of_strings[list_of_strings.index(string1) + 1 :]:

            # Convert the current string to lowercase
            string2_lower = string2.lower()

            # Check if the current strings are anagrams of each other
            if sorted(string1_lower) == sorted(string2_lower):

                # Increment the counter for anagram pairs
                pair_counter += 1

    # Return whether there are at most 131 pairs of anagrams in the given list
    return pair_counter <= 131
