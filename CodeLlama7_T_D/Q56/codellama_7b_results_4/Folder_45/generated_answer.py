
def all_substring_of_size_n(string, n):
    # Check if the given string and size are valid
    if len(string) < n or n < 1:
        return []

    # Initialize an empty list to store the substrings
    substrings = []

    # Iterate through each character in the string
    for i in range(len(string) - n + 1):

        # Check if the current substring is already present in the list
        if string[i:i+n] not in substrings:

            # Add the current substring to the list
            substrings.append(string[i:i+n])

    # Return the list of all distinct substrings of size n
    return substrings
