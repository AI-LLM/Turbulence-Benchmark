
def all_even_ints_exclusive(lst):
    even_ints = []
    for num in lst[0:5]:
        if num % 2 == 0:
            even_ints.append(num)
    return even_ints
