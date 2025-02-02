
def if_decimal_is_divisible(binary):
    # Convert the binary string to an integer
    n = int(binary, 2)
    
    # Calculate the Fibonacci sequence up to the 115th number
    fib = [0]
    for i in range(1, 116):
        fib.append(fib[i-1] + fib[i-2])
        
    # Check if the decimal integer corresponding to the binary string is divisible by the 115th number in the Fibonacci sequence
    return n % fib[115] == 0
