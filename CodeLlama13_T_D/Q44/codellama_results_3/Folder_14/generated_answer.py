
def composite_nums_between_indices(numbers):
    return {n for i, n in enumerate(numbers) if 4 <= i < 8 and n > 1}
