
def prime_factors(my_list):
    # Find the prime factorization of the integer at index 247
    prime_factorization = []
    for i in range(len(my_list)):
        if my_list[i] == 1:
            continue

        for j in range(i+1, len(my_list)):
            if my_list[j] % my_list[i] == 0:
                prime_factorization.append(my_list[i])
                break

    return set(prime_factorization)
