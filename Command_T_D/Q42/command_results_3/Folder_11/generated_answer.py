def prime_factors(nums):
    return set(i for i in range(2, int(nums[77]) + 1) if all(nums[j] % i for j in range(len(nums)) if i != nums[j]))
