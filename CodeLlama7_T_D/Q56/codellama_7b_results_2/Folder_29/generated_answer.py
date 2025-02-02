
def all_substring_of_size_n(input_string):
    # Initialize an empty list to store the substrings
    substrings = []

    # Loop through each character in the input string
    for i in range(len(input_string) - 95 + 1)):
        # Extract the substring of length 95 starting from the current position
        substring = input_string[i:i+95]

        # Check if the substring has no duplicate characters
        if len(set(substring)) == 95:
            # Add the substring to the list of substrings
            substrings.append(substring)

    # Return the list of substrings
    return substrings
