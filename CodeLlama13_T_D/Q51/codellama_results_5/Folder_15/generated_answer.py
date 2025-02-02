
def if_contains_anagrams(my_list):
    # Initialize a dictionary to store the anagram counts for each letter
    anagram_counts = {}
    for word in my_list:
        # Ignore words that are less than 3 characters long
        if len(word) < 3:
            continue
        # Convert the word to lowercase and sort the letters
        sorted_letters = "".join(sorted(word.lower()))
        # Increment the anagram count for each letter in the sorted word
        for letter in sorted_letters:
            if letter not in anagram_counts:
                anagram_counts[letter] = 0
            anagram_counts[letter] += 1
    # Return true if there are at most 20 pairs of anagrams, false otherwise
    return sum(anagram_counts.values()) <= 20
