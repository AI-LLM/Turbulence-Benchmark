
def all_substring_of_size_n(s):
    n = 70
    subs = []
    for i in range(len(s) - n + 1):
        sub = s[i:i+n]
        if len(sub) == n and not any(c in subs for c in sub):
            subs.append(sub)
    return subs
