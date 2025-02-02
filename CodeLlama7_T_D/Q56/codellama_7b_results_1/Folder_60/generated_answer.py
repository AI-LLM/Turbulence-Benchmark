
def all_substring_of_size_n(input_string):
    # Initialize an empty list to store the substrings
    substrings = []

    # Loop through each character in the input string
    for i in range(len(input_string) - 2)):
        # Extract the substring of length 36 starting at index i
        substring = input_string[i:i+36]

        # If the substring is not already in the list of substrings, add it to the list
        if substring not in substrings:
            substrings.append(substring)

    # Return the list of all distinct substrings of length 36 with no duplicate characters
    return substrings
