def find_sum_of_multiples(n):
    # Create a variable to store the sum of the multiples
    sum_of_multiples = 0
    # Create a variable to store the multiple of n
    multiple = 1
    # Create a variable to store the maximum multiple of n
    max_multiple = n * 366
    # Iterate from 1 to the maximum multiple of n
    for i in range(1, max_multiple + 1):
        # If i is a multiple of n, then add it to the sum of the multiples
        if i % n == 0:
            sum_of_multiples += i
        # Update the multiple of n
        multiple = i
    # Return the sum of the multiples
    return sum_of_multiples
