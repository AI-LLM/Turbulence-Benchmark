
def all_left_truncatable_prime(nums):
    primes = []
    for i in range(len(nums)):
        num = nums[i]
        if is_left_truncatable_prime(num, nums):
            primes.append(num)
    return sorted(primes)

def is_left_truncatable_prime(num, nums):
    for i in range(len(nums)):
        if num % nums[i] == 0:
            return False
    return True
