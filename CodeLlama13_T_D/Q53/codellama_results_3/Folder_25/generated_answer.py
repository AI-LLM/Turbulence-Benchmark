
def palindrome_of_length_at_least_n(s, n):
    return {w for w in s.lower().split() if len(w) >= n and w == w[::-1]}
