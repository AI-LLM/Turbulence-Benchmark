
def palindrome_of_length_n(string):
    result = set()
    for i in range(len(string) - 78 + 1):
        substr = string[i:i+78]
        if substr == substr[::-1]:
            result.add(substr)
    return result
