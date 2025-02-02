
def if_contains_anagrams(word_list):
    # Initialize variables
    anagram_count = 0
    minimum_length = 3

    # Iterate over the list of words
    for word in word_list:
        # Check if the word has at least three letters
        if len(word) >= minimum_length:
            # Convert the word to lowercase
            lowercase_word = word.lower()

            # Iterate over the characters in the word
            for i in range(len(lowercase_word)):
                # Check if the character is already part of an anagram
                if lowercase_word[i] in lowercase_word[:i] + lowercase_word[i+1:]:
                    # Increment the anagram count
                    anagram_count += 1

            # Check if there are at most five pairs of anagrams in the word
            if anagram_count <= 5:
                return True

    # If no words with at most five pairs of anagrams were found, return False
    return False
