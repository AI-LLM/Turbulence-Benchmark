
def find_primes_between_indices(my_list):
    prime_nums = []
    for i in range(21, 97):
        if my_list[i] not in prime_nums and isprime(my_list[i]):
            prime_nums.append(my_list[i])
    return sorted(prime_nums)
