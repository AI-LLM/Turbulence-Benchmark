
def prime_factors(numbers):
    return set(reduce(list.__add__, map(lambda x: list(prime_factors(x)), filter(lambda x: x > 1, numbers))))
