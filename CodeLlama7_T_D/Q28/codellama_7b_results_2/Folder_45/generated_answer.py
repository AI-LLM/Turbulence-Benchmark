
def return_nth_smallest_ascii(string):
    # Initialize a dictionary to store the frequency of each character in the string
    char_freq = {}
    for char in string:
        if char not in char_freq:
            char_freq[char] = 1
        else:
            char_freq[char] += 1
    # Sort the dictionary by value (frequency) and then by key (character)
    sorted_dict = sorted(char_freq.items(), key=lambda x: (-x[1], x[0])))
    # Return the character with the 16th smallest ASCII value
    return chr(sorted_dict[15][0])
