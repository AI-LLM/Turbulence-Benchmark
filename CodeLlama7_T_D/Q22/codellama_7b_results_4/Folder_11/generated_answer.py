
def find_sum_of_multiples(num):
    if num <= 0:
        raise ValueError("Number must be a positive integer.")

    sum = 0

    for i in range(1, 57):

        if i % num == 0:

            sum += i


    return sum
