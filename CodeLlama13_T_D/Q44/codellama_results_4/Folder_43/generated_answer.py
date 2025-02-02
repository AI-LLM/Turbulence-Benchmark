
def composite_nums_between_indices(my_list):
    indices = range(31, 72)
    return {x for x in my_list if is_composite(x) and index in indices}

def is_composite(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
