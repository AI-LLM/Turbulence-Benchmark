
def insert_before_character(string):
    result = ""
    for i in range(len(string)):
        if string[i] == "r":
            result += "1" + string[i]
        else:
            result += string[i]
    return result
