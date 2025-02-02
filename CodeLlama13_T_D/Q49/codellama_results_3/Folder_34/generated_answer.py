
def if_decimal_is_divisible(binary_string):
    binary_list = list(binary_string)
    decimal = 0
    for i in range(len(binary_list)):
        decimal += (2 ** (len(binary_list) - i - 1)) * int(binary_list[i])
    fibonacci_sequence = [0, 1]
    for i in range(2, 100):
        fibonacci_sequence.append(fibonacci_sequence[i-1] + fibonacci_sequence[i-2])
    return decimal % fibonacci_sequence[23] == 0
