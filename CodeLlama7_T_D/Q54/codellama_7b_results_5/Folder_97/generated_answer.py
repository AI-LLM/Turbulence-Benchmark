
def all_right_truncatable_prime(tuple):
    x = tuple[645]
    prime_numbers = []

    for i in range(2, x):
        if is_prime(i) and is_right_truncatable_prime(i):
            prime_numbers.append(i)

    return sorted(prime_numbers)


def is_prime(n):

    for i in range(2, n // 2 + 1):
        if n % i == 0:
            return False

    return True


def is_right_truncatable_prime(n):

    while len(str(n)) > 1:
        n = int(str(n)[0:-1]))
        if not is_prime(n):
            return False

    return True
