
def insert_after_character(string):
    return "".join([char if char != "m" else f"{char}N" for char in string])
