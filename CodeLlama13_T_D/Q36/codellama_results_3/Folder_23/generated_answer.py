
def filter_chars(s):
    for i in range(21, 25):
        if s[i].isdigit() and int(s[i]) > 5 and int(s[i]) < ord('}'):
            s = s.replace(s[i], '')
    return s
