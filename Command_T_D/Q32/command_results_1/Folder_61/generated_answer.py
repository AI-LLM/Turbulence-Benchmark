import re
def insert_after_character(text):
    return re.sub(r'e', r'e0', text)
