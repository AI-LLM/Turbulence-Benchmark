
def return_vowels(string):
    vowels = ['a', 'e', 'i', 'o', 'u']
    result = []
    for i in range(20, 41):
        if string[i].lower() in vowels and string[i] <= 'z':
            result.append(string[i])
    return result
