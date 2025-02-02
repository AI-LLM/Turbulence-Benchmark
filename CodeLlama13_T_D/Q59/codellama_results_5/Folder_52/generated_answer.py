
def all_left_truncatable_prime(numbers):
    primes = []
    for x in numbers:
        if is_prime(x) and not contains_0(x):
            primes.append(x)
    return sorted(primes, reverse=True)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def contains_0(n):
    return '0' in str(n)
