
def prime_factors(my_list):
    # Find the prime factor of the integer at index 849
    prime_factor = None
    for i in range(len(my_list)):
        if my_list[i] == 849:
            prime_factor = i

    return set([prime_factor])
