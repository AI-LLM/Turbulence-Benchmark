
def all_left_truncatable_prime(tuple):
    x = tuple[93]
    left_truncatable_primes = []
    for i in range(2, x):
        if is_prime(i) and is_left_truncatable_prime(i):
            left_truncatable_primes.append(i)
    return sorted(left_truncatable_primes)
def is_prime(n):
    if n < 2:
        return False

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    return True

def is_left_truncatable_prime(n):
    if str(n)[-1] != "1":
        return False

    for i in range(len(str(n)) - 2, -1, -1):
        if str(n)[i] == "0":
            return False

        if not is_prime(int(str(n)[i:] + str(n)[:i])):
            return False

    return True
