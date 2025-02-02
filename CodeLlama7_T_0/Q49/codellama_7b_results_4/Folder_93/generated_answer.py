
def if_decimal_is_divisible(binary_string):
    # Convert the binary string to an integer
    decimal_integer = int(binary_string, 2)
    # Calculate the 59th Fibonacci number
    fib_59 = (((1 + 5 ** 0.5) / 2) ** 59 - ((1 + 5 ** 0.5) / 2)) / (5 ** 0.5)
    # Check if the decimal integer is divisible by the 59th Fibonacci number
    return decimal_integer % fib_59 == 0
