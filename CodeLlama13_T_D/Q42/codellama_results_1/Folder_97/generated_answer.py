
def prime_factors(nums):
    if len(nums) == 0:
        return set()
    
    index = 132
    num = nums[index]
    
    factors = []
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            factors.append(i)
            
    return set(factors)
