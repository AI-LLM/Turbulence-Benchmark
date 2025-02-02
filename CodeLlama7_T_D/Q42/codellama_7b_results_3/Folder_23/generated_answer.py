
def prime_factors(my_list):
    # Check if the input is a list of positive integers
    if not all(isinstance(x, int) and x > 0 for x in my_list)):
        raise ValueError("Input must be a list of positive integers.")
    # Get the integer at index 69
    num = my_list[69]
    # Initialize an empty set to store prime factors
    prime_factors = set()
    # Loop through all numbers from 2 to the square root of num
    for i in range(2, int(num ** 0.5) + 1):
        # If i divides num without leaving a remainder, add it to the set of prime factors
        if num % i == 0:
            prime_factors.add(i)
            # Update num by dividing it by i
            num = int(num / i)
    # Return the set of all prime factors of the integer at index 69
    return prime_factors
