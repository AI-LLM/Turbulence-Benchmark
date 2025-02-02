
def all_substring_of_size_n(s, n):
    """
    Returns the list of all distinct substrings of length n in the given string s.
    If no such substrings exist in the given string, the function should return an empty list.
    The order of the substrings in the returned list does not matter.
    Args:
        s (str): The input string.
        n (int): The length of the substrings to find.
    Returns:
        list[str]: A list of all distinct substrings of length n in the given string s, or an empty list if no such substrings exist.
    """

    # Check if the input string is valid
    if not isinstance(s, str) or len(s) < 1:
        return []

    # Check if the substring length is valid
    if n < 1 or n > len(s):
        return []

    # Create a list to store the substrings
    substrings = []

    # Iterate over the characters in the input string
    for i in range(len(s) - n + 1):

        # Check if the substring is already in the list
        if s[i:i+n] not in substrings:

            # Add the substring to the list
            substrings.append(s[i:i+n])

    # Return the list of substrings
    return substrings
