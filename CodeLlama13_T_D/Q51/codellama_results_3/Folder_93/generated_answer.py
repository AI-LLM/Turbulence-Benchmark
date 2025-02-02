
def if_contains_anagrams(my_list):
    # Create a dictionary to store the anagram counts for each string
    anagram_counts = {}

    # Iterate through the list of strings and check if they are anagrams
    for s in my_list:
        # Convert the string to lowercase and sort it
        sorted_s = "".join(sorted(s.lower()))

        # Check if the sorted string is already in the dictionary
        if sorted_s not in anagram_counts:
            # If not, add it to the dictionary with a count of 1
            anagram_counts[sorted_s] = 1
        else:
            # If it is already in the dictionary, increment its count
            anagram_counts[sorted_s] += 1

    # Check if there are at most 36 pairs of anagrams in the list
    return len(anagram_counts) <= 36
