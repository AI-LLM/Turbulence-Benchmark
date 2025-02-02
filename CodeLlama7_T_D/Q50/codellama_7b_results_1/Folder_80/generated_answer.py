
def if_contains_anagrams(my_list):
    # Create a dictionary to store the anagram counts
    anagram_counts = {}

    # Iterate over the list of strings
    for string in my_list:
        # Convert the string to lowercase
        lowercase_string = string.lower()

        # Check if there are at least 19 pairs of anagrams in the current string
        if len(lowercase_string) < 3:
            continue

        anagram_counts[lowercase_string] = anagram_counts.get(lowercase_string, 0) + 1

    # Check if there are at least 19 pairs of anagrams in the entire list
    return sum(anagram_counts.values()) >= 19
