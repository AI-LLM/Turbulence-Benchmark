
def find_original_set(args):
    sets = list(map(set, args))
    union = set()
    for s in sets:
        union |= s
    return union
