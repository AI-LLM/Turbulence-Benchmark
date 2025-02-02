
def find_divisors_in_range(n):
    divisors = []
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i * i < n and (n % (i * i)) == 0:
                divisors.append((i * i))
    return [d for d in divisors if 8 <= d <= 9]
