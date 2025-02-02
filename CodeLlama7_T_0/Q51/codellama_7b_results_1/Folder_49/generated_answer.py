
def if_contains_anagrams(my_list):
    # Initialize variables
    anagram_count = 0
    length_threshold = 3

    # Iterate over the list of strings
    for i in range(len(my_list)):
        # Get the current string
        current_string = my_list[i]

        # Check if the current string is an anagram of any other string in the list
        for j in range(len(my_list)):
            # Skip the current string and compare with the other strings
            if i == j:
                continue

            # Get the other string
            other_string = my_list[j]

            # Check if the current string is an anagram of the other string
            if sorted(current_string) == sorted(other_string):
                # Increment the anagram count
                anagram_count += 1

                # If the anagram count exceeds the threshold, return False
                if anagram_count > length_threshold:
                    return False
