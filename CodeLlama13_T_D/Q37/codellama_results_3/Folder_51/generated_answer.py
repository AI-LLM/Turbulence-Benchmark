
def filter_chars(s):
    result = ''
    for i in range(len(s)):
        if 36 <= ord(s[i]) <= 40 and s[i] >= 'H' and s[i] <= 'e':
            continue
        result += s[i]
    return result
