
def sum_even_ints_inclusive(numbers):
    result = 0
    for i in range(29, 94):
        if numbers[i] % 2 == 0:
            result += numbers[i]
    return result
