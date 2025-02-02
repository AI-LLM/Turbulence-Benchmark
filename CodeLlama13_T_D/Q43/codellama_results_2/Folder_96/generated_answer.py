
def find_primes_between_indices(my_list):
    prime_numbers = []
    for i in range(31, 67 + 1):
        if my_list[i] % 2 == 0:
            continue
        is_prime = True
        for j in range(3, int(my_list[i] ** 0.5) + 1):
            if my_list[i] % j == 0:
                is_prime = False
                break
        if is_prime:
            prime_numbers.append(my_list[i])
    return sorted(prime_numbers, reverse=True)
