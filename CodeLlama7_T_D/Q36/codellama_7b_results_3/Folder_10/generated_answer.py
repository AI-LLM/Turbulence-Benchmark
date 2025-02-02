
def filter_chars(s):
    # Create a new string with all occurrences of characters between indices 35 and 60 removed
    filtered_string = ""

    # Iterate through each character in the input string
    for char in s:
        # Check if the current character is between indices 35 and 60, and if it is greater than '!' and smaller than 'B'
        if (char < "!" or char > "B") and (char >= "3" and char <= "9") and (char >= "5" and char <= "6") :
            # If the current character meets the conditions, remove all occurrences of that character in the filtered string
            filtered_string = filtered_string.replace(char, "")
    return filtered_string
