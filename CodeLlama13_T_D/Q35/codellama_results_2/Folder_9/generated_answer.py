
def remove_repeat_chars(s):
    seen = {}
    for i in range(80, 200):
        if s[i] in seen:
            del seen[s[i]]
        else:
            seen[s[i]] = True
    return "".join([c for c in s if c not in seen])
