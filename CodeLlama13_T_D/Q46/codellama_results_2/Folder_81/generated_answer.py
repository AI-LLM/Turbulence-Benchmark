
def gcf_three_nums(my_list):
    indices = [20, 17, 28]
    nums = [my_list[i] for i in indices]
    return get_gcf(nums)

def get_gcf(numbers):
    if len(numbers) == 2:
        a, b = numbers
        gcd = gcd_two(a, b)
        return gcd
    else:
        a, *rest = numbers
        gcd = gcd_one(a, rest)
        return gcd

def gcd_two(a, b):
    if a == 0:
        return b
    elif b == 0:
        return a
    else:
        while a != b:
            if a > b:
                a -= b
            else:
                b -= a
        return a

def gcd_one(a, numbers):
    for num in numbers:
        gcd = gcd_two(a, num)
        a = gcd
    return gcd
