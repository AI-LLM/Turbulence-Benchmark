import re
def filter_chars(s):
    return re.sub(r'(?<=.)([7-9]d?)', '', s)
