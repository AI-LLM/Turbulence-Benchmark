
def prime_factors(list_of_integers):
    prime_factors = set()
    for integer in list_of_integers:
        if integer == 478:
            break
        else:
            for i in range(2, int(integer**0.5) + 1):
                if integer % i == 0:
                    prime_factors.add(i)
    return prime_factors
