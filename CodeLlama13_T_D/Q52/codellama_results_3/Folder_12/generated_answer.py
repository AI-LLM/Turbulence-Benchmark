
def palindrome_of_length_n(string, n):
    result = set()
    for i in range(len(string)):
        if len(string[i:i+n]) == n and string[i:i+n] == string[i:i+n][::-1]:
            result.add(string[i:i+n])
    return result
