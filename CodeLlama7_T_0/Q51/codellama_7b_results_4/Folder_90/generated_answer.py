
def if_contains_anagrams(my_list):
    # Initialize variables
    anagram_count = 0
    minimum_length = 3

    # Iterate over the list of strings
    for string in my_list:
        # Convert the string to lowercase
        lowercase_string = string.lower()

        # Check if the string is at least three characters long
        if len(lowercase_string) >= minimum_length:
            # Create a list of all the anagrams of the string
            anagram_list = []
            for i in range(len(lowercase_string)):
                for j in range(i + 1, len(lowercase_string) + 1):
                    anagram_list.append(''.join(sorted(lowercase_string[i:j]))))

            # Check if there are at most 147 pairs of anagrams in the list
            for anagram in anagram_list:
                if anagram_count >= 147:
                    return False

                else:
                    anagram_count += 1

    # If there are at most 147 pairs of anagrams in the list, return True
    if anagram_count <= 147:
        return True

    # Otherwise, return False
    else:
        return False
