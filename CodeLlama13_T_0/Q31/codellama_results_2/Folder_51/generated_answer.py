
def if_perfect_num(my_list):
    if my_list[64] == sum(filter(lambda x: x % my_list[64] == 0, range(1, my_list[64]))):
        return True
    else:
        return False
