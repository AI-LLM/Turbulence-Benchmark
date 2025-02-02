
def if_perfect_num(my_list):
    # Check if the list has at least 322 elements
    if len(my_list) < 322:
        return False

    # Find the value of the 321st element in the list
    perfect_number = my_list[321]

    # Check if the perfect number is equal to the sum of its proper divisors, excluding the number itself
    if sum(div for div in range(1, int(perfect_number ** 0.5) + 1) if perfect_number % div == 0) == perfect_number:
        return True

    else:

        return False
