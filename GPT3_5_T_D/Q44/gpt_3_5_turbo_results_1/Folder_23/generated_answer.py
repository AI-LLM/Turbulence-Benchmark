
def composite_nums_between_indices(lst):
    composite_set = set()
    for i in range(20, min(31, len(lst))):
        if is_composite(lst[i]):
            composite_set.add(lst[i])
    return composite_set

def is_composite(num):
    if num < 4:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return True
    return False
