
def return_vowels(string):
    vowels = ['a', 'e', 'i', 'o', 'u']
    result = []
    for i in range(82, 90):
        if string[i] in vowels:
            result.append(string[i])
    return result
