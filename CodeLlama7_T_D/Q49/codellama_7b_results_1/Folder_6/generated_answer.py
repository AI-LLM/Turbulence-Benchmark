
def if_decimal_is_divisible(binary_rep):
    # Convert the binary representation to a decimal integer
    decimal_int = int(binary_rep, 2)
    # Initialize variables for Fibonacci sequence
    a, b = 0, 1
    # Loop through the Fibonacci sequence until the 185th number is reached
    while b < 185:
        a, b = b, a + b
    # Check if the decimal integer is divisible by the 185th number in the Fibonacci sequence
    return decimal_int % b == 0
