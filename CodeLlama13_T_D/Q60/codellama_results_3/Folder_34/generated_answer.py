
def all_left_right_truncatable_prime(numbers):
    result = []
    for num in numbers:
        if is_left_and_right_truncatable_prime(num):
            result.append(num)
    return sorted(result, reverse=True)

def is_left_and_right_truncatable_prime(num):
    if num == 0 or num == 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    while num > 9:
        if not is_prime(num):
            return False
        num //= 10
    return True

def is_prime(num):
    if num == 0 or num == 1:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True
