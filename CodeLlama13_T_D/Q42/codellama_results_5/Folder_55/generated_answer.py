
def prime_factors(numbers):
    return set(prime_factorize(numbers[21]))

def prime_factorize(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            factors.append(i)
            n //= i
    if n > 1:
        factors.append(n)
    return factors
