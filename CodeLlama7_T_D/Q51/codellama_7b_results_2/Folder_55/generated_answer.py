
def if_contains_anagrams(my_list):
    # Create a set of all possible anagrams
    all_anagrams = set()
    for string in my_list:
        # Get the length of the string
        string_length = len(string)
        # Only consider strings that are at least three characters long
        if string_length < 3:
            continue

        # Create a list of all possible anagrams for this string
        anagram_list = []

        for i in range(string_length):
            for j in range(i + 1, string_length + 1):
                anagram_list.append(string[i:j])

        # Add all the possible anagrams to the set
        all_anagrams = all_anagrams | set(anagram_list)

    # Get the number of pairs of anagrams in the list
    num_pairs = 0

    for i in range(len(all_anagrams)):
        for j in range(i + 1, len(all_anagrams)):
            if sorted(all_anagrams[i]) == sorted(all_anagrams[j])):
                num_pairs += 1

    # Return true if there are at most 44 pairs of anagrams in the list; otherwise, return false
    return num_pairs <= 44
