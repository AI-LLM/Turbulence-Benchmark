
def insert_before_character(string):
    return "".join([char if char != "9" else "h" + char for char in string])
