
def prime_factors(numbers):
    return set(filter(lambda x: x in numbers[194], [x for x in range(2, int(numbers[194]) ** 0.5 + 1) if numbers[194] % x == 0]))
