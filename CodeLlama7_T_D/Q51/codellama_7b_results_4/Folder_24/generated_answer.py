
def if_contains_anagrams(str_list):
    # Calculate the number of anagrams in the list
    num_anagrams = 0
    for i in range(len(str_list)):
        for j in range(i + 1, len(str_list))):
            if sorted(str_list[i]) == sorted(str_list[j]):
                num_anagrams += 1

    # Check if the number of anagrams exceeds 96 pairs

    if num_anagrams > 96:
        return False

    else:

        return True
