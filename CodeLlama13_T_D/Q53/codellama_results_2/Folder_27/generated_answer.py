
def palindrome_of_length_at_least_n(s):
    return {p for p in s if len(p) >= 100 and p == p[::-1]}
