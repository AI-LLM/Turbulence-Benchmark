
def return_vowels(string):
    vowels = []
    for i in range(34, 69):
        char = string[i]
        if char == '_' or char > 'o':
            continue

        else:

            vowels.append(char)

    return vowels
