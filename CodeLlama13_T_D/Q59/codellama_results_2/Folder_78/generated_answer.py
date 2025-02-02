
def all_left_truncatable_prime(nums):
    x = nums[23]
    prime_list = []
    for i in range(10, x+1):
        if is_prime(i) and not '0' in str(i):
            prime_list.append(i)
    return sorted(prime_list)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True
