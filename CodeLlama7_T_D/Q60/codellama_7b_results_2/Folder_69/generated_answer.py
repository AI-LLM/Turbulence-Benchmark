
def all_left_right_truncatable_prime(x):
    # Initialize an empty list to store the prime numbers
    prime_numbers = []

    # Iterate from 1 to x
    for i in range(1, x + 1):

        # Check if i is a prime number
        if is_prime(i):

            # Check if i is left-and-right-truncatable prime number
            if is_left_right_truncatable_prime(i):

                # Add i to the list of prime numbers
                prime_numbers.append(i)

    # Sort the list of prime numbers in descending order
    prime_numbers.sort(reverse=True)

    # Return the list of prime numbers
    return prime_numbers

def is_prime(n):

    # Check if n is greater than 1
    if n > 1:

        # Iterate from 2 to the square root of n
        for i in range(2, int(n ** 0.5) + 1):

            # Check if i divides n
            if n % i == 0:

                # Return False if i divides n
                return False

    # If n is greater than 1, it is a prime number
    return True

def is_left_right_truncatable_prime(n):

    # Check if n is a prime number
    if not is_prime(n):

        # Return False if n is not a prime number
        return False

    # Check if the leading leftmost and last rightmost digits of n are simultaneously successively removed down to a one- or two-digit prime.
    for i in range(len(str(n))) - 1):

        # Check if removing the leading leftmost digit from n results in a prime number
        if is_prime(int(str(n)[i:])))

            # Check if removing the last rightmost digit from n results in a prime number
            if is_prime(int(str(n)[:-i])), return True

    # If no left-and-right-truncatable prime number exists in the specified range, the function should return an empty list.
    return False
