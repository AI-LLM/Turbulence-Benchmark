
def return_vowels(string):
    vowels = ["a", "e", "i", "o", "u"]
    return_list = []
    for i in range(2, 7):
        if string[i] in vowels and string[i] > "^" and string[i] <= "d":
            return_list.append(string[i])
    return return_list
