
def filter_chars(s):
    s = list(s)
    for i in range(52, 89):
        if ord(s[i]) > ord('&') and ord(s[i]) < ord('c'):
            s.remove(s[i])
    return ''.join(s)
