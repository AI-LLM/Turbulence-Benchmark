
def prime_factors(lst):
    num = lst[74]
    factors = set()
    
    while num % 2 == 0:
        factors.add(2)
        num /= 2
    
    i = 3
    while i <= int(num**0.5) + 1:
        if num % i == 0:
            factors.add(i)
            num /= i
        else:
            i += 2
    
    if num > 1:
        factors.add(num)
    
    return factors
