
def return_vowels(s):
    vowels = ["a", "e", "i", "o", "u"]
    result = []
    for i in range(149, 313):
        if s[i].upper() in vowels and s[i].upper() <= "j":
            result.append(s[i])
    return result
