
def insert_after_character(string):
    result = ""
    for i in range(len(string)):
        if string[i] == 'E':
            result += string[i] + "I"
        else:
            result += string[i]
    return result
