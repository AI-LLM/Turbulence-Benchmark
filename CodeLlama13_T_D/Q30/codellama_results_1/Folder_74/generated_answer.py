
def insert_before_character(string, char):
    return "".join([char + s if s == char else s for s in string])
