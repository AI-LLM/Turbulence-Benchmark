
def if_perfect_num(my_list):
    # Check if the list is empty or has less than 78 elements
    if len(my_list) < 78:
        return False

    # Get the integer at index 77
    num = my_list[77]

    # Check if the integer is a perfect number
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            sum = i + num / i

            if sum == num:
                return True

    return False
