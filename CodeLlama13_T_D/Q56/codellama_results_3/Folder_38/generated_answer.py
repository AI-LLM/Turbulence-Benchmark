
def all_substring_of_size_n(string):
    substrings = []
    for i in range(len(string) - 34):
        substring = string[i:i+35]
        if len(set(substring)) == 35 and not any(substring in s for s in substrings):
            substrings.append(substring)
    return substrings
