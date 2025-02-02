
import math

def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def find_primes_between_indices(lst):
    prime_numbers = []
    for i in range(23, 40):
        if is_prime(lst[i]):
            prime_numbers.append(lst[i])
    return sorted(prime_numbers)
