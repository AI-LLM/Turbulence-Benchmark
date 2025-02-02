
def if_contains_anagrams(wordlist):
    # Initialize counter for number of anagram pairs found
    pair_count = 0

    # Iterate over the word list and compare each word with its reverse
    for word in wordlist:
        rev_word = ''.join(reversed(word))

        # If the word is a palindrome, it is already an anagram of itself
        if word == rev_word:
            continue

        # Check if the reverse of the word is also in the list
        if rev_word in wordlist:
            # Increment counter for number of anagram pairs found
            pair_count += 1

    # Return true if there are at least 277 pairs of anagrams in the given list; otherwise, return false
    return pair_count >= 277
