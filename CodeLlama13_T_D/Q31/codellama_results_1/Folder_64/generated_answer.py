
def if_perfect_num(my_list):
    n = my_list[1]
    s = sum(range(2, int(n**0.5) + 1))
    return True if n == s else False
