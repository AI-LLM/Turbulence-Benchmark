
def find_divisors_in_range(n):
    divisors = []
    for i in range(40, 75):
        if n % i == 0 and i >= 40 and i <= 74:
            divisors.append(i)
    return divisors
