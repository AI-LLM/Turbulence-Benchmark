
def remove_repeat_chars(s):
    # Find all characters that occur more than once between indices 51 and 76, exclusive
    char_count = {}
    for i in range(50, 76):
        if s[i] not in char_count:
            char_count[s[i]] = 1
        else:
            char_count[s[i]] += 1

    # Remove all occurrences of characters that occur more than once between indices 51 and 76, exclusive
    for i in range(len(s)):
        if s[i] in char_count and char_count[s[i]] > 1:
            s = s[:i] + s[i+1:]
            break

    return s
