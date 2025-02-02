
def all_left_right_truncatable_prime(x):
    # Initialize an empty list to store the prime numbers
    prime_numbers = []

    # Iterate from 1 to x (inclusive)
    for i in range(1, x + 1):
        # Check if i is a prime number
        if is_prime(i):
            # Check if i is left-and-right-truncatable prime number
            if is_left_right_truncatable_prime(i):
                # Add i to the list of prime numbers
                prime_numbers.append(i)

    # Sort the list of prime numbers in ascending order
    prime_numbers.sort()

    return prime_numbers

# Function to check if a number is a prime number
def is_prime(n):
    # Check if n is greater than 1
    if n > 1:
        # Iterate from 2 to the square root of n
        for i in range(2, int(n ** 0.5) + 1):
            # Check if i divides n
            if n % i == 0:
                # If i divides n, return False
                return False

    # If n is greater than 1 and has no factors other than 1 and itself, return True
    return True

# Function to check if a number is a left-and-right-truncatable prime number
def is_left_right_truncatable_prime(n):
    # Check if n is a prime number
    if is_prime(n):
        # Initialize an empty list to store the digits of n
        digits = []

        # Get the digits of n in reverse order
        while n > 0:
            digits.append(n % 10)
            n //= 10

        # Check if the leading leftmost and last rightmost digits of n are simultaneously successively removed down to a one- or two-digit prime
        for i in range(len(digits)):
            # Get the number obtained by removing the leading leftmost digit from n
            num = int(''.join(map(str, digits[i:]))))

            # Check if num is a prime number
            if is_prime(num):
                # If num is a prime number, return True
                return True

        # If the leading leftmost and last rightmost digits of n are simultaneously successively removed down to a one- or two-digit prime, return False
        return False

    # If n is not a prime number, return False
    return False
