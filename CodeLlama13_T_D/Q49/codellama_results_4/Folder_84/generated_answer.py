
def if_decimal_is_divisible(binary_string):
    # Convert the binary string to an integer
    decimal = int(binary_string, 2)
    
    # Get the 196th number that occurs in the Fibonacci sequence
    fibonacci_num = [0] + [a + b for a, b in zip([0] + [0], [0] + [1])]
    divisor = fibonacci_num[197]
    
    # Check if the decimal is divisible by the divisor
    return decimal % divisor == 0
