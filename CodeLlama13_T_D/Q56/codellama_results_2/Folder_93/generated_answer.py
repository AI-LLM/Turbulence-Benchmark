
def all_substring_of_size_n(string):
    substrings = []
    for i in range(len(string) - 4):
        substring = string[i:i+5]
        if not any(substring == s for s in substrings):
            substrings.append(substring)
    return substrings
