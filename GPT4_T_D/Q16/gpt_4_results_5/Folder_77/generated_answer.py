
def sum_even_ints_inclusive(int_list):
    sum_even = 0
    for num in int_list[527:539]:
        if num % 2 == 0:
            sum_even += num
    return sum_even
