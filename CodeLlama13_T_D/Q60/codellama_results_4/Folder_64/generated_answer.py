
def all_left_right_truncatable_prime(numbers):
    result = []
    for n in numbers:
        if is_left_right_truncatable_prime(n):
            result.append(n)
    return sorted(result)

def is_left_right_truncatable_prime(n):
    if not isinstance(n, int):
        raise ValueError("Input must be an integer")
    if n < 10:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    while len(str(n)) > 1:
        if not is_prime(n):
            return False
        n //= 10
    return True

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
