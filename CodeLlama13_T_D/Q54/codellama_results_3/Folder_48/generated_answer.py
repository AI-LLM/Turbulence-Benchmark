
def all_right_truncatable_prime(numbers):
    primes = []
    for number in numbers:
        if is_right_truncatable_prime(number):
            primes.append(number)
    return sorted(primes)

def is_right_truncatable_prime(n):
    if n < 10:
        return False
    for i in range(1, int(len(str(n)) / 2)):
        if not is_prime(n // (10 ** i)):
            return False
    return True

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
