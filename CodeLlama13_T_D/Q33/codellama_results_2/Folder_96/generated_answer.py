
def return_vowels(string):
    vowels = []
    for i in range(77, 84):
        if string[i].isalpha() and string[i] >= 'g' and string[i] <= 'G':
            vowels.append(string[i])
    return vowels
