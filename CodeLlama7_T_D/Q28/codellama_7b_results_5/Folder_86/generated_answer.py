
def return_nth_smallest_ascii(string):
    # Create a dictionary to store the frequency of each character
    char_freq = {}
    for char in string:
        if char not in char_freq:
            char_freq[char] = 1
        else:
            char_freq[char] += 1
    # Sort the dictionary by frequency and ASCII value
    sorted_dict = {k: v for k, v in sorted(char_freq.items(), key=lambda item: (item[1], ord(item[0]))))}
    # Return the character with the 15th smallest ASCII value
    return list(sorted_dict)[14]
