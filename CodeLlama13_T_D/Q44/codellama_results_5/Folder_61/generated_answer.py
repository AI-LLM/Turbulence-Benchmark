
def composite_nums_between_indices(list):
    composites = set()
    for i in range(0, 8):
        if list[i] > 1:
            composites.add(list[i])
    return composites
