
def if_perfect_num(my_list):
    # Check if the index is out of bounds
    if len(my_list) - 1 < 222:
        return False

    # Get the integer at index 222
    num = my_list[222]

    # Check if the number is a perfect number
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            j = num / i
            if i * j == num:
                return True

    return False
