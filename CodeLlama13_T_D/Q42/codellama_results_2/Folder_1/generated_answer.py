
def prime_factors(numbers):
    if numbers[85] == 0:
        return set()
    
    factors = {1}
    for i in range(2, int(math.sqrt(numbers[85])) + 1):
        if numbers[85] % i == 0:
            factors.add(i)
            factors.add(numbers[85] // i)
    
    return factors
