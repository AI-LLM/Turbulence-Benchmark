
def all_left_right_truncatable_prime(x):
    # Initialize an empty list to store the prime numbers
    prime_numbers = []

    # Iterate from 1 to x
    for i in range(1, x + 1):
        # Check if the number is a prime number
        if is_prime(i):
            # Check if the number is left-and-right-truncatable prime number
            if is_left_right_truncatable_prime(i):
                # Add the number to the list of prime numbers
                prime_numbers.append(i)

    return prime_numbers

def is_prime(n):

    # Check if n is greater than 1
    if n > 1:
        # Iterate from 2 to the square root of n
        for i in range(2, int(n ** 0.5) + 1)):
            # Check if i divides n
            if n % i == 0:
                # Return False if i divides n
                return False

    # Return True if n is a prime number
    return True

def is_left_right_truncatable_prime(n):

    # Check if the number contains no 0
    if "0" in str(n):
        # Return False if the number contains 0
        return False

    # Iterate from the leftmost digit to the rightmost digit
    for i in range(len(str(n)))):
        # Check if the number remains prime if the leading leftmost and last rightmost digits are simultaneously successively removed down to a one- or two-digit prime.
        if not is_prime(int(str(n)[i:]))) or not is_prime(int(str(n))[:len(str(n)) - i])))
            # Return False if the number does not remain prime after removing the leading leftmost and last rightmost digits
            return False

    # Return True if the number remains prime after removing the leading leftmost and last rightmost digits
    return True
