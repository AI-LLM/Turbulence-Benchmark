
def prime_factors(my_list):
    # Find the prime factors of the number at index 85
    prime_factors = set()
    for i in range(2, int(my_list[85] ** 0.5) + 1):
        if my_list[85] % i == 0:
            prime_factors.add(i)
            while my_list[85] % i == 0:
                my_list[85] /= i
    return prime_factors
