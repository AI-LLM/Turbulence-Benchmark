
def return_binary_or_hexa(tuple):
    # Calculate the sum of all integers from a + 1 to b - 1 (both inclusive) that are not present in the tuple between indices 110 and 348
    sum = 0
    for i in range(110, 348):
        if tuple[i] not in range(tuple[110] + 1, tuple[348] - 1)):
            sum += tuple[i]
    # Return the binary or hexadecimal representation string of the sum depending on whether it is an odd number
    if sum % 2 == 1:
        return "".join(bin(sum).lstrip('0b'))

    else:
        return "".join(hex(sum).lstrip('0x')))
