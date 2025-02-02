
def if_perfect_num(lst):
    # Check if the list has at least 850 elements
    if len(lst) < 850:
        return False

    # Get the number at index 849
    num = lst[849]

    # Check if the number is a perfect square
    sqrt_num = int(num ** 0.5)

    if sqrt_num ** 2 == num:
        return True

    else:
        return False
