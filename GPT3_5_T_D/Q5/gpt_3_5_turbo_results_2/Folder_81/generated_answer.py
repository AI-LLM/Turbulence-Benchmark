
def all_neg_ints_inclusive(lst):
    result = []
    for num in lst[10:101]:
        if num < 0:
            result.append(num)
    return result
