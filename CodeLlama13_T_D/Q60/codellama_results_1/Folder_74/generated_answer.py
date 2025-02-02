
def all_left_right_truncatable_prime(x):
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    primes = []
    for i in range(x+1):
        if is_prime(i): primes.append(i)

    left_truncatable_primes = []
    for p in primes:
        n = len(str(p)) - 1
        while n > 0 and p % (10**n) == 0: n -= 1
        if n > 0: left_truncatable_primes.append(p)

    right_truncatable_primes = []
    for p in primes:
        n = len(str(p)) - 1
        while n > 0 and p % (10**n) == 0: n -= 1
        if n > 0: right_truncatable_primes.append(p)

    left_right_truncatable_primes = []
    for l in left_truncatable_primes:
        for r in right_truncatable_primes:
            if l == r and l != 0: left_right_truncatable_primes.append(l)

    return sorted(left_right_truncatable_primes)
