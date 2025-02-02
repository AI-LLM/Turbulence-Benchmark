
def gcf_two_nums(my_list):
    # Find the greatest common factor of two numbers
    # using the Euclidean algorithm

    def euclid(a, b):
        while b:
            a, b = b, a % b
        return a

    num1 = my_list[64]
    num2 = my_list[28]
    gcf = euclid(num1, num2)
    return gcf
