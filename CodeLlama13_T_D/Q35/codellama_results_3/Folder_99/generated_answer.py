
def remove_repeat_chars(string):
    seen = {}
    result = ""
    for i, char in enumerate(string):
        if i < 450 or i > 905:
            continue
        if char not in seen:
            seen[char] = True
            result += char
    return result
