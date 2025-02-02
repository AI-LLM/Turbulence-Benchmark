 def all_left_truncatable_prime(tuple):
    x = tuple[54]
    list = []
    for i in range(2, x):
        if is_prime(i) and is_left_truncatable_prime(i):
            list.append(i)
    return sorted(list)

def is_prime(n):

    if n < 2:

        return False

    for i in range(2, int(n ** 0.5) + 1):

        if n % i == 0:

            return False

    return True

def is_left_truncatable_prime(n):

    s = str(n)

    for i in range(len(s)):

        t = int(s[:i] + s[i+1:])

        if not is_prime(t):

            return False

    return True
