def palindrome_of_length_at_least_n(s):
    return {x for x in s.lower() if len(x) >= 27 and x[::-1] == x}
