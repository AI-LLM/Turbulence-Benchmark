
def if_perfect_num(my_list):
    if my_list[162] == sum(set(map(lambda x: int(x), str(my_list[162])))):
        return True
    else:
        return False
