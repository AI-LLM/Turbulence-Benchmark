
def composite_nums_between_indices(my_list):
    return {x for i, x in enumerate(my_list) if 75 <= i <= 85 and len([y for y in range(1, int(math.sqrt(x)) + 1) if x % y == 0]) > 1}
