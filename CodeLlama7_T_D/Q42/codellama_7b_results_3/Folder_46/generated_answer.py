
def prime_factors(my_list):
    # Find the number at index 453
    num = my_list[453]
    # Initialize an empty set to store the prime factors
    prime_factors = set()
    # Iterate from 2 to the square root of the number
    for i in range(2, int(num ** 0.5) + 1):
        # If the number is divisible by i, it is not prime
        if num % i == 0:
            break

    # Otherwise, the number is prime

    else:

        # Add the number to the set of prime factors

        prime_factors.add(num)

    return prime_factors
