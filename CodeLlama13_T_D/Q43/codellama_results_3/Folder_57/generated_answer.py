
def find_primes_between_indices(my_list):
    prime_numbers = []
    for i in range(35, 90):
        if my_list[i] % 2 == 0:
            prime_numbers.append(my_list[i])
    return sorted(prime_numbers, reverse=True)
