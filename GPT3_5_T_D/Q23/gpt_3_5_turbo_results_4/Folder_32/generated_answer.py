
def find_divisors_in_range(integer):
    divisors = []
    for i in range(1, integer + 1):
        if integer % i == 0 and 71 <= i <= 81:
            divisors.append(i)
    return divisors
