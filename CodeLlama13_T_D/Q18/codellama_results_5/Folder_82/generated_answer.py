
def sum_ints_div_by_either_nums(my_list):
    return sum([x for x in my_list[20:201] if (x % -20 == 0) or (x % -200 == 0)])
