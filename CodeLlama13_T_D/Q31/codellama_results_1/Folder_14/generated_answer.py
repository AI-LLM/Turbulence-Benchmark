
def if_perfect_num(my_list):
    if len(my_list) <= 17:
        return False
    
    num = my_list[17]
    divisors = set()
    for i in range(1, int(num**0.5) + 1):
        if num % i == 0:
            divisors.add(i)
            if num // i != i:
                divisors.add(num // i)
    
    return sum(divisors) == num
