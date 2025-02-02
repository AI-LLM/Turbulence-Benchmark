
def if_decimal_is_divisible(binary_string):
    # Convert the binary string to an integer
    decimal_int = int(binary_string, 2)
    # Initialize variables for the Fibonacci sequence
    fib1 = 0
    fib2 = 1
    # Calculate the 49th number in the Fibonacci sequence
    for i in range(48):
        fib1, fib2 = fib2, fib1 + fib2
    # Check if the decimal integer is divisible by the 49th number in the Fibonacci sequence
    if decimal_int % fib2 == 0:
        return True

    else:

        return False
