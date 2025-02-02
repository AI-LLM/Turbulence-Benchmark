
def all_left_right_truncatable_prime(x):
    # Initialize an empty list to store the prime numbers
    prime_numbers = []

    # Iterate from 1 to x (inclusive)
    for i in range(1, x + 1):

        # Check if i is a prime number
        if is_prime(i):

            # Check if the leading leftmost and last rightmost digits are simultaneously successively removed down to a one- or two-digit prime
            if is_left_right_truncatable_prime(i):

                # Add i to the list of prime numbers
                prime_numbers.append(i)

    # Sort the list of prime numbers in ascending order
    prime_numbers.sort()

    # Return the list of prime numbers
    return prime_numbers

# Define a function to check if a number is prime
def is_prime(n):

    # Check if n is greater than 1
    if n > 1:

        # Initialize a variable to store the remainder
        remainder = 0

        # Iterate from 2 to the square root of n
        for i in range(2, int(n ** 0.5) + 1):

            # Check if i divides n without leaving a remainder
            if n % i == 0:

                # Set the remainder to 1
                remainder = 1

                # Break out of the loop
                break

        # Return False if the remainder is non-zero
        return remainder == 0

    # Return True if n is less than or equal to 1
    else:

        # Return True
        return True

# Define a function to check if a number is left-and-right-truncatable prime
def is_left_right_truncatable_prime(n):

    # Check if n is less than or equal to 1
    if n <= 1:

        # Return False
        return False

    # Initialize a variable to store the number of digits in n
    num_digits = len(str(n))

    # Iterate from 2 to the square root of n
    for i in range(2, int(n ** 0.5) + 1):

        # Check if i divides n without leaving a remainder
        if n % i == 0:

            # If the leading leftmost and last rightmost digits are not equal to each other, return False
            if str(n)[-num_digits:] != str(n)[:num_digits]:

                # Return False
                return False

    # Return True
    return True
