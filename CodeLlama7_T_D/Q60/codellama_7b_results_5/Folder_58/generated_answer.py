 def all_left_right_truncatable_prime(num):
    # Initialize an empty list to store the prime numbers
    primes = []

    # Check if num is a prime number

    if is_prime(num):
        # Add num to the list of prime numbers

        primes.append(num)

    # Loop through all the integers from 2 to x (the last index of num)

    for i in range(2, int(num)):

        # Check if i is a prime number and if it contains no 0

        if is_prime(i) and '0' not in str(i):

            # Add i to the list of prime numbers

            primes.append(i)

    # Sort the list of prime numbers in descending order

    primes = sorted(primes, reverse=True)

    # Return the list of prime numbers

    return primes
