
def return_vowels(string):
    vowels = []
    for i in range(43, 70):
        if string[i].lower() in "aeiou" and string[i] <= "b":
            vowels.append(string[i])
    return vowels
