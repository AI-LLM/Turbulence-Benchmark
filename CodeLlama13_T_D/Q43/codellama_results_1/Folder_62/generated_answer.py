
def find_primes_between_indices(lst):
    primes = []
    for i in range(29, 98):
        if lst[i] in (2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89):
            primes.append(lst[i])
    return sorted(primes, reverse=True)
