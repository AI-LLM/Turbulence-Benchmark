
def if_perfect_num(my_list):
    return my_list[478] == sum(filter(lambda x: x % my_list[478] == 0, range(1, my_list[478])))
