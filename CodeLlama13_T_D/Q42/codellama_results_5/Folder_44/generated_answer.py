
def prime_factors(numbers):
    return set([int(i) for i in str(numbers[828]) if int(i) > 1 and numbers[828] % int(i) == 0])
