
def if_contains_anagrams(strings):
    # Check if there are at least three pairs of anagrams in the given list
    for i in range(len(strings) - 1):
        for j in range(i + 1, len(strings)):
            # If the strings are anagrams, return True
            if sorted(strings[i]) == sorted(strings[j]):
                return True

    # If there are no pairs of anagrams, return False

    return False
