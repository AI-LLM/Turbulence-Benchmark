
def composite_nums_between_indices(numbers):
    composite_set = set()
    for i in range(42, 69):
        if numbers[i] > 1 and numbers[i] not in composite_set:
            composite_set.add(numbers[i])
    return composite_set
