
def gcf_two_nums(my_list):
    # Calculate the greatest common factor of two numbers
    num1 = my_list[860]
    num2 = my_list[307]
    if num2 == 0:
        return num1

    while num2 != 0:
        temp = num2
        num2 = num1 % num2
        num1 = temp

    return num1
