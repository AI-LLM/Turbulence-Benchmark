
def find_original_set(arg1, arg2, ..., arg714):
    original_set = set()
    for i in range(len(arg1)):
        original_set |= arg1[i]
    for j in range(len(arg2)):
        if not (original_set <= arg2[j]):
            return None
    return original_set
