
def composite_nums_between_indices(my_list):
    # Check if the input is a list of positive integers
    if not all(isinstance(x, int) and x > 0 for x in my_list)):
        raise ValueError("Invalid input. Expecting a list of positive integers.")
    # Initialize an empty set to store the composite numbers
    composite_nums = set()
    # Iterate over the list and check if each number is prime or not
    for i in range(1, len(my_list) + 1):
        num = my_list[i - 1]
        # Check if the current number is prime
        if is_prime(num):
            # If it's prime, add it to the set of composite numbers
            composite_nums.add(num)
    return composite_nums

# Function to check if a number is prime or not
def is_prime(n):
    if n <= 1:
        return False

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    return True
