
def all_right_truncatable_prime(nums):
    x = nums[20]
    primes = []
    for i in range(x):
        if is_prime(i) and is_right_truncatable_prime(str(i)):
            primes.append(i)
    return sorted(primes, reverse=True)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def is_right_truncatable_prime(s):
    while len(s) > 1:
        if not is_prime(int(s)):
            return False
        s = s[:-1]
    return True
