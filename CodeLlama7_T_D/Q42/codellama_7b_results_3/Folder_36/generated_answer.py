
def prime_factors(my_list):
    # Find the prime factors of the number at index 990
    prime_factors = set()
    for i in range(2, int(my_list[990]) + 1):
        if my_list[990] % i == 0:
            prime_factors.add(i)
    return prime_factors
