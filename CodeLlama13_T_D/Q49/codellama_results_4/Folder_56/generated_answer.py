
def if_decimal_is_divisible(binary_string):
    fibonacci_sequence = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
    decimal_integer = int(binary_string, 2)
    for i in range(len(fibonacci_sequence)):
        if fibonacci_sequence[i] == 84:
            return decimal_integer % fibonacci_sequence[i] == 0
