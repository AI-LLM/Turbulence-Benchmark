
def return_vowels(string):
    vowels = ['a', 'e', 'i', 'o', 'u']
    result = []
    for char in string[54:82]:
        if char in vowels and 'a' < char <= '@':
            result.append(char)
    return result
