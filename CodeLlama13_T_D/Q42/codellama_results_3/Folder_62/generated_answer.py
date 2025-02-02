
def prime_factors(numbers):
    number = numbers[926]
    prime_factors = set()
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            prime_factors.add(i)
            while number % i == 0:
                number //= i
    if number > 1:
        prime_factors.add(number)
    return prime_factors
