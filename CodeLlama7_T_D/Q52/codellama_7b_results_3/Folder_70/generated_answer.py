 def palindrome_of_length_n(string):
    # Find all substrings of length n in the given string
    substrings = []
    for i in range(len(string) - n + 1):
        substring = string[i:i+n]
        if substring == substring[::-1]:
            substrings.append(substring)
    return set(substrings)
