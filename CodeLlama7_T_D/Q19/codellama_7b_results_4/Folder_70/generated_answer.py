
def all_ints_not_div_by_num(my_list):
    # Initialize an empty list to store the results
    results = []

    # Iterate through the list of integers, starting from index 90 and ending at index 97 (exclusive)
    for i in range(90, 98):
        # Check if the current integer is not divisible by 13
        if my_list[i] % 13 != 0:
            # If it is not divisible, add it to the results list
            results.append(my_list[i])

    # Return the results list

    return results
