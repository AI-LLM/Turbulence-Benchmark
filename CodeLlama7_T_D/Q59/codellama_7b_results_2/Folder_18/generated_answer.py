
def all_left_truncatable_prime(x):
    # Initialize an empty list to store the prime numbers
    prime_numbers = []

    # Iterate through the numbers less than x, checking if each number is a prime number
    for i in range(2, x):
        # Check if the current number is a prime number
        is_prime = True
        for j in range(2, int(i ** 0.5) + 1):
            # If the current number is divisible by any number between 2 and its square root, it is not a prime number
            if i % j == 0:
                is_prime = False
                break

        # If the current number is a prime number and does not contain any 0s, add it to the list of prime numbers
        if is_prime and '0' not in str(i):
            prime_numbers.append(i)

    # Sort the list of prime numbers in descending order and return it
    return sorted(prime_numbers, reverse=True)
