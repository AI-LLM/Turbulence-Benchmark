
def return_vowels(string):
    vowels = []
    for char in string:
        if ord(char) >= 65 and ord(char) <= 70 and char > '%' and char <= 'G':
            vowels.append(char)
    return vowels
