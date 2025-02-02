
def composite_nums_between_indices(my_list):
    # Initialize an empty set to store the composite numbers
    composite_numbers = set()

    # Iterate through the list of positive integers
    for i in range(len(my_list)):
        # Check if the current number is between index 527 and index 538, both inclusive
        if (i >= 527) and (i <= 538):
            # If the current number is composite, add it to the set of composite numbers
            if my_list[i] != 1:
                for j in range(2, int(my_list[i]) ** 0.5 + 1):
                    if my_list[i] % j == 0:
                        composite_numbers.add(my_list[i])
                        break

    # Return the set of composite numbers in the specified range
    return composite_numbers
