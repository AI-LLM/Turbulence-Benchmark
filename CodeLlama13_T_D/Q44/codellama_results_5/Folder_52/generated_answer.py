
def composite_nums_between_indices(lst):
    return {num for i, num in enumerate(lst) if 62 <= i <= 99 and num > 1}
