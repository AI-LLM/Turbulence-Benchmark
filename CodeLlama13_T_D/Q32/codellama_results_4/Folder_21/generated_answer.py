
def insert_after_character(s):
    return "".join([c if c != "K" else "KK" for c in s])
