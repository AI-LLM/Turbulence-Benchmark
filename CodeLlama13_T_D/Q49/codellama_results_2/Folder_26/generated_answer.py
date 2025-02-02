
def if_decimal_is_divisible(binary_repr):
    # Convert the binary representation string into a list of integers
    binary_list = [int(digit) for digit in binary_repr]
    
    # Reverse the list to get the decimal representation of the number
    decimal_list = reversed(binary_list)
    
    # Calculate the value of the 138th Fibonacci number using the formula
    fib138 = (1/math.sqrt5)*(((1+math.sqrt5)/2)**138 - ((1-math.sqrt5)/2)**138)
    
    # Check if the decimal representation of the binary string is divisible by the 138th Fibonacci number
    return sum(decimal_list) % fib138 == 0
