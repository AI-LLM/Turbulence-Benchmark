
def palindrome_of_length_at_least_n(string, n):
    return {pal for pal in re.findall(r'(?i)[a-zA-Z]{%s}' % (n+1), string) if pal == pal[::-1]}
