
def all_right_truncatable_prime(n):
    # Initialize an empty list to store the prime numbers
    prime_list = []

    # Iterate through the number n and check if it is a prime number
    for i in range(2, n + 1):

        # Check if the number is divisible by any number between 2 and sqrt(n)
        for j in range(2, int(i ** 0.5) + 1)):
            if i % j == 0:
                break

        else:

            # If the number is prime, add it to the list of prime numbers
            prime_list.append(i)

    # Return the sorted list of all right-truncatable prime numbers less than n
    return sorted(prime_list, reverse=True)
