
def return_vowels(string):
    vowels = []
    for i in range(32, 97):
        if string[i].isalpha() and string[i] > 'Q' and string[i] <= ']':
            vowels.append(string[i])
    return vowels
