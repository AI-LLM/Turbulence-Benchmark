
def all_left_right_truncatable_prime(x):
    # Initialize an empty list to store the prime numbers
    prime_numbers = []
    
    # Iterate from 1 to x (both inclusive)
    for i in range(1, x + 1):
        # Check if the number is a prime number
        if is_prime(i):
            # Check if the number is left-and-right-truncatable prime number
            if is_left_right_truncatable_prime(i):
                # Add the number to the list of prime numbers
                prime_numbers.append(i)
    
    # Return the sorted list of prime numbers
    return sorted(prime_numbers)
```
Note that this function uses two helper functions: `is_prime` and `is_left_right_truncatable_prime`. These functions are not included in the code above, but they can be implemented as follows:
```
def is_prime(n):
    # Check if n is a positive integer
    if n <= 0 or type(n) != int:
        return False
    
    # Check if n is divisible by any number between 1 and sqrt(n)
    for i in range(2, int(sqrt(n))) + 1):
        if n % i == 0:
            return False
    
    return True
```

```
def is_left_right_truncatable_prime(n):
    # Check if n is a prime number
    if not is_prime(n):
        return False
    
    # Check if the leading leftmost and last rightmost digits of n are simultaneously successively removed down to a one- or two-digit prime.
    for i in range(1, len(str(n)))):
        temp = int(str(n)[i:] + str(n)[:i])
        if not is_prime(temp):
            return False
    
    return True
