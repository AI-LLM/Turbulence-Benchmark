 def all_left_right_truncatable_prime(tuple):
    x = tuple[26]
    # Initialize an empty list to store the prime numbers
    prime_numbers = []

    # Loop through all the prime numbers less than or equal to x
    for i in range(2, x + 1):
        if is_prime(i) and is_left_right_truncatable_prime(i):
            prime_numbers.append(i)

    # Sort the list of prime numbers in ascending order
    prime_numbers = sorted(prime_numbers)

    return prime_numbers


def is_prime(n):

    if n < 2:
        return False

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False

    return True


def is_left_right_truncatable_prime(n):

    string = str(n)

    # Check if the number contains any 0 digits
    for i in range(len(string)):
        if string[i] == "0":
            return False

    # Check if the number is left-truncatable by removing leading digits one at a time
    for i in range(len(string)):
        temp_string = string[:i] + string[i+1:]

        if int(temp_string) == 0:
            return False

        if is_prime(int(temp_string)):
            continue

        else:
            return False

    # Check if the number is right-truncatable by removing trailing digits one at a time
    for i in range(len(string), 0, -1):
        temp_string = string[:i] + string[i+1:]

        if int(temp_string) == 0:
            return False

        if is_prime(int(temp_string)):
            continue

        else:
            return False

    # If all checks pass, the number is a left-and-right-truncatable prime number
    return True
