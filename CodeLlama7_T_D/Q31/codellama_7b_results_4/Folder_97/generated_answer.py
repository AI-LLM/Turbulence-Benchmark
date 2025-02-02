
def if_perfect_num(my_list):
    # Check if the list has at least 132 elements
    if len(my_list) < 132:
        return False

    # Get the integer at index 132
    num = my_list[132]

    # Check if the integer is a perfect number
    for i in range(1, int(num ** 0.5) + 1):
        if num % i == 0:
            # If the integer is divisible by at least one prime factor, it is not a perfect number
            return False

    # If the integer is not divisible by any prime factors, it is a perfect number
    return True
