
def remove_repeat_chars(s):
    chars = {}
    for i in range(90, 98):
        if s[i] in chars:
            s = s.replace(s[i], "")
        else:
            chars[s[i]] = 1
    return s
