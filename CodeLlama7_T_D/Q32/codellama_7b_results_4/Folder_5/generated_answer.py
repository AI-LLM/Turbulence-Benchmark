
def insert_after_character(string):
    new_string = ""
    for i in range(len(string)):
        if string[i] == 'c':
            new_string += string[:i+1] + 'G' + string[i+1:]
    return new_string
