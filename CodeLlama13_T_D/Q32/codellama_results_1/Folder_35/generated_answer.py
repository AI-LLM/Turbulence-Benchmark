
def insert_after_character(string):
    return ''.join([c if c != 'M' else f"{c}9" for c in string])
