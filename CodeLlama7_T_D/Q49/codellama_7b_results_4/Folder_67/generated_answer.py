
def if_decimal_is_divisible(binary_string):
    # convert binary string to integer
    decimal_integer = int(binary_string, 2)
    # find the position of the most significant digit in the Fibonacci sequence
    msb_position = len(binary_string) - (binary_string.rindex('1') + 1))
    # calculate the 107th number that occurs in the Fibonacci sequence
    fibonacci_sequence = [0, 1]
    for i in range(2, msb_position):
        fibonacci_sequence.append(fibonacci_sequence[i - 1] + fibonacci_sequence[i])
    # check if the decimal integer is divisible by the 107th number that occurs in the Fibonacci sequence
    return (decimal_integer % fibonacci_sequence[msb_position - 1]) == 0
