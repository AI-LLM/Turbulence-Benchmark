
def palindrome_of_length_at_least_n(string, n):
    return {pal for pal in set(re.findall("[a-zA-Z]{" + str(n) + ",}", string)) if pal == pal[::-1]}
