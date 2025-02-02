
def if_contains_anagrams(my_list):
    # Initialize a counter for the number of anagrams found
    num_anagrams = 0

    # Iterate over the list of strings
    for string in my_list:
        # Convert the string to lowercase
        lowercase_string = string.lower()

        # Check if the string is an anagram of another string in the list
        for other_string in my_list:
            # If the string and the other string are not the same, check if they are anagrams of each other
            if string != other_string:
                # Convert the other string to lowercase
                lowercase_other_string = other_string.lower()
                # Check if the string is a permutation of the other string
                if sorted(lowercase_string) == sorted(lowercase_other_string):
                    # Increment the counter for the number of anagrams found
                    num_anagrams += 1

        # If there are more than 173 pairs of anagrams in the list, return False
        if num_anagrams > 173:
            return False

    # Return True if there are at most 173 pairs of anagrams in the list
    return True
