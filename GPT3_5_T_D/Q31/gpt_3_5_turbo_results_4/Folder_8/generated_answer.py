
def if_perfect_num(lst):
    def is_perfect_num(n):
        factors = []
        for i in range(1, n):
            if n % i == 0:
                factors.append(i)
        if sum(factors) == n:
            return True
        else:
            return False
    
    num = lst[49]
    return is_perfect_num(num)
