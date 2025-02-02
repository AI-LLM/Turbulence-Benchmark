
def all_left_truncatable_prime(tuple):
    x = tuple[84]
    left_truncatable_primes = []
    for i in range(2, x):
        if is_prime(i) and left_truncatable_prime(i):
            left_truncatable_primes.append(i)
    return sorted(left_truncatable_primes, reverse=True)
def is_prime(n):
    if n < 2:
        return False

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    return True

def left_truncatable_prime(n):
    if not is_prime(n):
        return False

    digits = [int(digit) for digit in str(n)]

    for i in range(len(digits)):
        if digits[i] == 0:
            return False

        new_n = int(''.join(map(str, digits[i:])))

        if not is_prime(new_n):
            return False

    return True
