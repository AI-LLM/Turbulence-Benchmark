
def all_ints_div_by_both_two_nums(my_list):
    # Initialize an empty list to store the divisible integers
    divisible_integers = []

    # Iterate through the list starting from index 42

    for i in range(42, len(my_list)):
        # Check if the current integer is divisible by both 15 and 87

        if my_list[i] % 15 == 0 and my_list[i] % 87 == 0:
            # If it is, add it to the list of divisible integers

            divisible_integers.append(my_list[i])

    # Return the list of all integers divisible by both 15 and 87 from index 42 to index 43, both inclusive. If no such integers exist in the specified range, the function should return an empty list.

    return divisible_integers
