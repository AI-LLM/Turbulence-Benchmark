
def sum_odd_ints_inclusive(my_list):
    odd_nums = []
    for i in range(64, 67):
        if my_list[i] % 2 != 0:
            odd_nums.append(my_list[i])
    return sum(odd_nums) or 0
