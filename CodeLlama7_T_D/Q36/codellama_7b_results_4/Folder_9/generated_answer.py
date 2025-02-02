
def filter_chars(s):
    # Initialize an empty string to store the filtered characters
    filtered = ""

    # Iterate over each character in the input string
    for c in s:

        # If the current character is between the indices 50 and 92 (both exclusive) and it is not 'A' or 'Q', add it to the filtered string
        if (c > "5" and c < "9") and (c != "A" and c != "Q"):
            filtered += c

    # Return the filtered string
    return filtered
