def prime_factors(nums):
    return set(i for i in range(2, int(nums[83])+1) if all(nums[83] % i == 0 for j in range(2, int(nums[83])+1) if i*j != i*j))
