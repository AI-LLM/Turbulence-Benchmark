
def return_n_greatest_chars(s):
    return sorted([*set(s)], reverse=True)[:90]
