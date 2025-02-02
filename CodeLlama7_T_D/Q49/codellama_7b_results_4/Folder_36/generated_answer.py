
def if_decimal_is_divisible(binary_representation):
    # Convert the binary representation to an integer
    decimal_integer = int(binary_representation, 2)
    # Calculate the first 150 numbers of the Fibonacci sequence
    fibonacci_sequence = [0] * 150
    fibonacci_sequence[0] = 0
    for i in range(1, 150):
        fibonacci_sequence[i] = fibonacci_sequence[i - 1] + fibonacci_sequence[i - 2]
    # Check if the decimal integer is divisible by any number of the Fibonacci sequence
    for i in range(len(fibonacci_sequence)):
        if decimal_integer % fibonacci_sequence[i] == 0:
            return True

    return False
