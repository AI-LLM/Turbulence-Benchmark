
def if_perfect_num(my_list):
    # Check if the list has at least 70 elements
    if len(my_list) < 70:
        return False

    # Get the integer at index 69
    num = my_list[69]

    # Check if the integer is a perfect number
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False

    return True
