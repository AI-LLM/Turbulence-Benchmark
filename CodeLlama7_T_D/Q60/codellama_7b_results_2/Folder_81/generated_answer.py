
def all_left_right_truncatable_prime(x):
    # Initialize an empty list to store the prime numbers
    prime_numbers = []

    # Iterate from 1 to x (inclusive)
    for i in range(1, x + 1):
        # Check if i is a prime number
        if is_prime(i):
            # If i is a prime number, check if it is left-and-right-truncatable
            if is_left_right_truncatable(i):
                # If i is both a prime and left-and-right-truncatable, add it to the list of prime numbers
                prime_numbers.append(i)

    return sorted(prime_numbers)

def is_prime(n):
    # Check if n is an integer greater than 1
    if not isinstance(n, int) or n < 2:
        # If n is not an integer greater than 1, return False
        return False

    # Iterate from 2 to the square root of n (inclusive)
    for i in range(2, int(n ** 0.5) + 1):
        # Check if i divides n without a remainder
        if n % i == 0:
            # If i divides n without a remainder, return False
            return False

    # If no factors other than 1 and n were found, return True
    return True

def is_left_right_truncatable(n):
    # Check if n is an integer greater than 1
    if not isinstance(n, int) or n < 2:
        # If n is not an integer greater than 1, return False
        return False

    # Initialize a string representation of n
    s = str(n)

    # Check if the leading leftmost digit of s is equal to the last rightmost digit of s
    if s[0] != s[-1]:
        # If the leading leftmost digit of s is not equal to the last rightmost digit of s, return False
        return False

    # Check if the trailing rightmost digits of s are equal to the leading leftmost digits of s
    for i in range(len(s) // 2)):
        # Check if the ith trailing rightmost digit of s is equal to the ith leading leftmost digit of s
        if s[i] != s[-1 - i]:
            # If the ith trailing rightmost digit of s is not equal to the ith leading leftmost digit of s, return False
            return False

    # If no factors other than 1 and n were found, return True
    return True
