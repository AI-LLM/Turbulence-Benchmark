
def gcf_two_nums(my_list):
    # Calculate the greatest common factor of the two numbers at indices 7 and 9
    num1 = my_list[7]
    num2 = my_list[9]
    if num1 == num2:
        return num1

    else:

        for i in range(min(num1, num2), max(num1, num2)) + 1):
            if num1 % i == 0 and num2 % i == 0:
                return i

        return 1
