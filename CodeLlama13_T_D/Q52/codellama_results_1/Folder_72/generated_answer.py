
def palindrome_of_length_n(s, n):
    return {p for p in (s[i:i+n] for i in range(len(s) - n + 1)) if p == p[::-1]}
