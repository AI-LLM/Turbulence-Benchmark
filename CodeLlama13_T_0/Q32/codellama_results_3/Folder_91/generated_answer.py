
def insert_after_character(string):
    return "".join([char if char != "a" else f"{char}0" for char in string])
