
def find_original_set(*args):
    sets = list(args)
    original_set = set()
    for s in sets:
        original_set |= s
    return original_set
