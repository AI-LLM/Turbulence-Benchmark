
def composite_nums_between_indices(lst):
    return {x for i, x in enumerate(lst) if i >= 30 and i <= 30 and x > 1}
