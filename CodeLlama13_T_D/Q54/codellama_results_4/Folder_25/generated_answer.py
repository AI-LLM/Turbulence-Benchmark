
def all_right_truncatable_prime(numbers):
    prime_numbers = []
    for num in numbers:
        if is_right_truncatable_prime(num):
            prime_numbers.append(num)
    return sorted(prime_numbers)

def is_right_truncatable_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
