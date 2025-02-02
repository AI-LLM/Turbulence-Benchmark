
def return_nth_smallest_ascii(string):
    # Initialize a dictionary to keep track of the ASCII values and their frequencies
    ascii_values = {}

    # Iterate through the string and update the ASCII value frequency dictionary
    for char in string:
        if char in ascii_values:
            ascii_values[char] += 1
        else:
            ascii_values[char] = 1

    # Sort the ASCII value frequency dictionary by key (ASCII value)
    sorted_ascii_values = sorted(ascii_values.items(), key=lambda x: x[0])

    # Return the 15th smallest ASCII value in the sorted list
    return sorted_ascii_values[14][0]
