
def if_decimal_is_divisible(binary_repr):
    # Convert the binary representation string into an integer
    decimal = int(binary_repr, 2)
    
    # Calculate the Fibonacci sequence up to the 117th number
    fibonacci = [0] + [i * (j+1) for i, j in enumerate([0, 1])]
    
    # Check if the decimal integer is divisible by the 117th Fibonacci number
    return decimal % fibonacci[117] == 0
