
def all_right_truncatable_prime(tuple):
    x = tuple[70]
    prime_numbers = []

    for i in range(2, x + 1):
        if is_prime(i) and is_right_truncatable_prime(i):
            prime_numbers.append(i)

    return sorted(prime_numbers, reverse=True)

def is_prime(n):

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    return True


def is_right_truncatable_prime(n):

    while len(str(n)) > 1:
        n = int(str(n)[0:-1]))

        if not is_prime(n):
            return False


    return True
