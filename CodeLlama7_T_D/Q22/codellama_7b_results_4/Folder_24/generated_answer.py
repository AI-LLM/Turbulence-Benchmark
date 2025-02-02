
def find_sum_of_multiples(num):
    if num < 0:
        raise ValueError("Input must be a positive integer.")
    sum = 0
    for i in range(1, 50):
        sum += (i * num)
    return sum
