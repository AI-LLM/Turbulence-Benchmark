
def insert_after_character(string):
    return "".join([char + "j" if char == "N" else char for char in string])
