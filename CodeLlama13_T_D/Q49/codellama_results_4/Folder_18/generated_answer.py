
def if_decimal_is_divisible(binary_repr):
    fibonacci_seq = [0, 1]
    for i in range(2, len(fibonacci_seq)):
        fibonacci_seq.append(fibonacci_seq[i-1] + fibonacci_seq[i-2])
    
    decimal_repr = int(binary_repr, 2)
    if decimal_repr % fibonacci_seq[198] == 0:
        return True
    else:
        return False
