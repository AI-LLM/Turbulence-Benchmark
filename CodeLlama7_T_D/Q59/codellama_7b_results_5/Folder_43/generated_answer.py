
def all_left_truncatable_prime(tuple):
    # Initialize a list to store the left-truncatable prime numbers
    prime_list = []
    # Iterate over the tuple of positive integers
    for i in range(len(tuple)):
        # Get the integer at index i
        x = tuple[i]
        # Check if x is a prime number
        if is_prime(x):
            # If x is a prime number, then add it to the list of left-truncatable prime numbers
            prime_list.append(x)
    # Return the sorted list of left-truncatable prime numbers less than x
    return sorted(prime_list, reverse=True)
# Define a function to check if a number is prime
def is_prime(n):
    # Check if n is greater than 1
    if n > 1:
        # Iterate from 2 to the square root of n
        for i in range(2, int(n ** 0.5) + 1)):
            # Check if i divides n
            if n % i == 0:
                # If i divides n, then return False
                return False

    # If n is less than or equal to 1, then return False
    else:
        return False
