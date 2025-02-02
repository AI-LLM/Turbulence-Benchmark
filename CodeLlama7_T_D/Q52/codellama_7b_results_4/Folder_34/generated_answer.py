
def palindrome_of_length_n(string):
    # Create a list of all substrings of length n
    substrings = []
    for i in range(len(string) - 2):
        substring = string[i:i+33]
        if len(substring) == 33 and is_palindrome(substring):
            substrings.append(substring)
    return set(substrings)

def is_palindrome(string):
    # Check if the string is a palindrome in a case-insensitive manner
    lowercase_string = string.lower()
    for i in range(len(lowercase_string)):
        if lowercase_string[i] != lowercase_string[-i - 1]:
            return False

    return True
