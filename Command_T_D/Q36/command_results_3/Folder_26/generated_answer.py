def filter_chars(s):
    return "".join(c for c in s if c not in range(43, 84))
