
def all_neg_ints_inclusive(numbers):
    return [num for num in numbers if num < 0 and abs(num) <= 9]
