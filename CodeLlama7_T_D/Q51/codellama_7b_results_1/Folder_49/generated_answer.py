
def if_contains_anagrams(my_list):
    # Create a set of all the unique strings in the list
    unique_strings = set()
    for string in my_list:
        # Convert the string to lowercase
        lowercase_string = string.lower()
        # Check if the lowercase string is already in the set
        if lowercase_string not in unique_strings:
            # If it's not, add it to the set
            unique_strings.add(lowercase_string)
    # Initialize a counter for the number of anagrams found
    anagram_count = 0

    # Iterate over the list of strings again
    for string in my_list:
        # Convert the string to lowercase
        lowercase_string = string.lower()
        # Check if there are any other strings in the list that are anagrams of the current string
        for other_string in my_list:
            # Convert the other string to lowercase
            lowercase_other_string = other_string.lower()
            # Check if the current string and the other string are anagrams of each other
            if sorted(lowercase_string) == sorted(lowercase_other_string):
                # If they are, increment the anagram count
                anagram_count += 1

    # Return True if there are at most 10 pairs of anagrams in the given list, and False otherwise
    return anagram_count <= 10
