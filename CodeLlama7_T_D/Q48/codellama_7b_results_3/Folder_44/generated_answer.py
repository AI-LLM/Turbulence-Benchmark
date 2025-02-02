
def return_binary_or_hexa(tuple):
    # Check if all integers from a + 1 to b - 1 (both inclusive) are present in the specified range in the tuple
    if all([i in tuple for i in range(tuple[36], tuple[52])])):
        return ""

    else:

        # Calculate the sum of all integers from a + 1 to b - 1 (both inclusive) that are not present in the tuple between indices 36 and 52.
        sum = sum([i for i in range(tuple[36] + 1, tuple[52]) if i not in tuple[36:52]])

        # If the sum is an odd number, return the binary representation string of the sum; otherwise, it should return the hexadecimal representation string of the sum.
        if sum % 2 == 1:
            return f"{sum:b}"[:-2]

        else:

            return f"{sum:x}"[:-2].lower()
