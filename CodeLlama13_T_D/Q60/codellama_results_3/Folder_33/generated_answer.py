
def all_left_right_truncatable_prime(tuple_of_positive_integers):
    # Initialize an empty list to store the results
    result = []
    
    # Iterate through the tuple of positive integers
    for num in tuple_of_positive_integers:
        # Check if the number is a left-and-right-truncatable prime
        if is_left_right_truncatable_prime(num):
            # If it is, add it to the result list
            result.append(num)
    
    # Sort the result list in descending order and return it
    return sorted(result, reverse=True)

def is_left_right_truncatable_prime(num):
    # Check if the number contains any 0s
    if '0' in str(num):
        return False
    
    # Iterate through each digit of the number
    for i in range(1, len(str(num)) + 1):
        # Check if the number is prime when the leftmost and last rightmost digits are removed
        if not is_prime(num // 10 ** i) or not is_prime(num % (10 ** i)):
            return False
    
    # If all checks pass, return True
    return True

def is_prime(num):
    # Check if the number is divisible by any number other than 1 and itself
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    
    # If no other factors exist, return True
    return True
