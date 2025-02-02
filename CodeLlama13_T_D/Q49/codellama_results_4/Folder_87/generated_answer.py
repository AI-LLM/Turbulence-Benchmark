
def if_decimal_is_divisible(binary_string):
    binary_list = binary_string.split()
    decimal_value = 0
    for i, digit in enumerate(binary_list):
        decimal_value += (2 ** i) * int(digit)
    fibonacci_sequence = [0] + [0, 1]
    for i in range(3, 500):
        fibonacci_sequence.append(fibonacci_sequence[i - 1] + fibonacci_sequence[i - 2])
    if decimal_value % fibonacci_sequence[180] == 0:
        return True
    else:
        return False
