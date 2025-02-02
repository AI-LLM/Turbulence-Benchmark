
def if_decimal_is_divisible(binary_string):
    # Convert the binary string to an integer
    decimal = int(binary_string, 2)
    
    # Calculate the 120th number in the Fibonacci sequence
    fibonacci_number = (1 + 5**0.5)/2
    
    # Check if the decimal is divisible by the 120th number
    return decimal % fibonacci_number == 0
