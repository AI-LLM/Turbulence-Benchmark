
def find_sum_of_multiples(num):
    if num < 0:
        raise ValueError("Invalid input. Positive integer expected.")

    sum = 0

    for i in range(1, 446):

        sum += num * i


    return sum
