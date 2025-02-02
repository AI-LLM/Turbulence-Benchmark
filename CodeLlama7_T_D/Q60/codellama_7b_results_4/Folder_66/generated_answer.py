
def all_left_right_truncatable_prime(tuple):
    x = tuple[25]
    left_right_truncatable_primes = []
    for i in range(1, x + 1):
        if is_prime(i) and is_left_right_truncatable_prime(i):
            left_right_truncatable_primes.append(i)
    return sorted(left_right_truncatable_primes, reverse=True)
def is_prime(n):
    if n < 2:
        return False

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    return True

def is_left_right_truncatable_prime(n):
    if not is_prime(n):
        return False

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    while n > 10:
        n = n // 10

    return n == 1 or n == 3 or n == 7 or n == 9
