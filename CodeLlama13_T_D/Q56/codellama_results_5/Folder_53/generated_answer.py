
def all_substring_of_size_n(s):
    # Initialize an empty list to store the substrings
    substrings = []
    
    # Loop through each substring of length 25 in the given string
    for i in range(len(s) - 24):
        # Check if the current substring has no duplicate characters
        if len(set(s[i:i+25])) == 25:
            # If it does, add it to the list of substrings
            substrings.append(s[i:i+25])
    
    # Return the list of substrings
    return substrings
