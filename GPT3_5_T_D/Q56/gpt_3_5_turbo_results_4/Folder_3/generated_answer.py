
def all_substring_of_size_n(string):
    substrings = []
    n = 93

    if len(string) < n:
        return substrings

    for i in range(len(string) - n + 1):
        substring = string[i : i + n]
        if len(set(substring)) == len(substring):
            substrings.append(substring)

    return list(set(substrings))
