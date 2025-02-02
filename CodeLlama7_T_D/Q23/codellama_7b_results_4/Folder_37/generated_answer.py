
def find_divisors_in_range(n):
    # Initialize an empty list to store the divisors
    divisors = []

    # Iterate from 1 to n-1, checking for divisibility
    for i in range(1, n):

        # If the current number is a divisor of 'n', add it to the list
        if n % i == 0:
            divisors.append(i)

    # Return the list of divisors between 3 and 7, inclusive
    return [divisor for divisor in divisors if 3 <= divisor <= 7]
