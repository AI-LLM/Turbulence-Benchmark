
def all_left_truncatable_prime(x):
    # Initialize a list to store the left-truncatable prime numbers less than x
    prime_list = []
    # Loop through all the numbers from 1 to x
    for i in range(1, x + 1):
        # Check if the number is a prime number
        if is_prime(i):
            # Initialize a string to store the number in string format
            num_str = str(i)

            # Loop through all the digits of the number from right to left
            for j in range(len(num_str) - 1, -1, -1):
                # Check if the digit is 0
                if num_str[j] == "0":
                    # Break out of the loop since no further digits need to be checked
                    break

                # Check if removing the leading leftmost digit makes the resulting number a prime number
                elif is_prime(int(num_str[:j] + num_str[j + 1:])):
                    # Add the number to the list of left-truncatable prime numbers
                    prime_list.append(i)

            # Sort the list of left-truncatable prime numbers in ascending order
            prime_list.sort()

    # Return the list of all left-truncatable prime numbers less than x
    return prime_list

# Define a function to check if a number is a prime number
def is_prime(n):
    # Check if n is an integer greater than 1
    if not (type(n) == int and n > 1):
        # Return False since n is not a prime number
        return False

    # Initialize a variable to store the value of n - 1
    i = n - 1

    # Loop through all the factors of n - 1
    while True:
        # Check if the factor is a prime number
        if is_prime(i):
            # Return False since n is not a prime number
            return False

        # If the factor is not a prime number, then check if it is equal to 1
        elif i == 1:
            # Return True since n is a prime number
            return True
