
def remove_repeat_chars(s):
    for i in range(2, 5):
        if s[i] != s[i-1] and s[i] != s[i+1]:
            continue
        else:
            while s.count(s[i]) > 1:
                s = s.replace(s[i], "")
    return s
