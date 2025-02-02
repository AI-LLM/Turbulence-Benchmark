
def find_divisors_in_range(n):
    # Initialize an empty list to store the divisors
    divisors = []

    # Iterate from 1 to n-1, checking if each number is a divisor of n
    for i in range(1, n):
        if n % i == 0:
            # If the current number is a divisor of n, add it to the list of divisors
            divisors.append(i)

    # Return the list of divisors that are between 7 and 9, both inclusive
    return [divisor for divisor in divisors if 7 <= divisor <= 9]
